// Local Linux observer for Judo's dynamically linked mj_step. Read only after the solve.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mujoco/mujoco.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void (*step_impl)(const mjModel*, mjData*);
static atomic_int count;
static int object, finger, wrist, floor_geom, live;
static double samples[2][11];  // ground xyz, hand xyz, other xyz, hand active, other impulse

void observe_setup(int obj, int f, int w, int g) {
  step_impl = dlsym(RTLD_NEXT, "mj_step");
  if (!step_impl) abort();
  object = obj; finger = f; wrist = w; floor_geom = g;
}
void observe_begin(int execution) {
  atomic_store(&count, 0);
  live = execution;
  memset(samples, 0, sizeof(samples));
}
int observe_end(double *output) {
  live = 0;
  if (output) memcpy(output, samples, sizeof(samples));
  return atomic_load(&count);
}
void mj_step(const mjModel* m, mjData* d) {
  if (!step_impl) step_impl = dlsym(RTLD_NEXT, "mj_step");
  step_impl(m, d);
  int k = atomic_fetch_add_explicit(&count, 1, memory_order_relaxed);
  if (!live) return;
  if (k >= 2) abort();
  double *row = samples[k];
  for (int i = 0; i < d->ncon; ++i) {
    const mjContact *c = d->contact + i;
    if (c->efc_address < 0) continue;
    int b0 = m->geom_bodyid[c->geom[0]], b1 = m->geom_bodyid[c->geom[1]];
    if ((b0 == object) == (b1 == object)) continue;
    int side = b1 == object ? 0 : 1;
    int other = m->geom_bodyid[c->geom[side]];
    int offset = c->geom[side] == floor_geom ? 0 : (other == wrist || other == finger ? 3 : 6);
    double wrench[6], world[3];
    mj_contactForce(m, d, i, wrench);
    mju_mulMatTVec3(world, c->frame, wrench);
    for (int j = 0; j < 3; ++j) row[offset+j] += (side ? -1 : 1) * world[j];
    if (offset == 3) row[9] = 1.;
    if (offset == 6) row[10] += mju_norm3(world) * m->opt.timestep;
  }
}
