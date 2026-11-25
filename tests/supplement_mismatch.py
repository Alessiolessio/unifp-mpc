"""Paired nominal prediction with lower ground traction or higher torso inertia."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from box_mpc import run
from box_scenario import generate, episode_seconds
from check_common_score import evaluate
from cone_mpc import generate_cone, cone_model, run_cone
from final_scores import write
from final_trials import CountPool, digest
import unifp_native
from unifp_native import Rollouts, World

ROOT = Path("artifacts/supplemental/uncertainty")
POLICY = "artifacts/encoder_actor.onnx"
CASES = [
    ("dev","box",4,"easy",17), ("dev","cone",0,"near",43),
    ("held","box",5501,"easy",451), ("held","box",5502,"medium",451),
    ("held","box",5503,"crossing",451), ("held","cone",5511,"near",451),
    ("held","cone",5512,"varied",451), ("held","cone",5513,"general",451),
]
PLANTS = {"nominal":{}, "ground":{"ground_scale":.55}, "torso":{"torso_scale":1.20}}


def validate(scene, plant, params):
    m, p = (mujoco.MjModel.from_binary_path(str(f)) for f in (scene,plant))
    for field, key, idx, groups in [
            ("body_mass","torso_scale",m.body("body").id,None),
            ("body_inertia","torso_scale",m.body("body").id,None),
            ("geom_friction","ground_scale",m.geom("ground").id,0)]:
        values = getattr(m,field).copy()
        if groups is None:
            values[idx] *= params.get(key,1.)
        else:
            values[idx,groups] *= params.get(key,1.)
        np.testing.assert_array_equal(values,getattr(p,field))
    for k in ["names","actuator_trnid","actuator_user","jnt_qposadr","jnt_dofadr","pair_friction",
              "geom_priority","geom_solref","geom_solimp","key_qpos","body_ipos"]:
        np.testing.assert_array_equal(getattr(m,k),getattr(p,k))
    # Ground outranks the feet. Box-floor uses an explicit pair; the cone outranks ground.
    ground = p.geom("ground")
    for i in range(p.ngeom):
        g = p.geom(i)
        if g.name and "foot_collision" in g.name:
            assert ground.priority.item() > g.priority.item()
        if g.name and "traffic_cone" in g.name and g.contype.item():
            assert g.priority.item() > ground.priority.item()
    if "box_body" in [p.body(i).name for i in range(p.nbody)]:
        assert p.pair("box_ground").id >= 0


def trial(task, tier, case, seed, settings, scene, plant, params, informed, arm, out):
    prediction = plant if informed else scene
    pool = Rollouts(World(str(prediction),POLICY),8)
    live = pool.execution_world(**({} if informed else params))
    assert pool.mismatched == (bool(params) and not informed)
    if not plant.exists():
        live.save_model(str(plant))
    validate(scene,plant,params)
    pool = CountPool(pool)
    out.mkdir(parents=True,exist_ok=True)
    if not (out/"scene.mjb").exists():
        (out/"scene.mjb").symlink_to(os.path.relpath(plant,out))
    args = SimpleNamespace(iterations=1,workers=8,seed=seed,controller="mobile",
                           policy=POLICY,out=out,warmup_ee=None,objective="upright_buffer",
                           horizon=1.,sumo_sampling=True,force_axes="z",
                           ee_force={"A":"off","B":"conditioned","C":"matched"}[arm],
                           candidates=16 if task=="cone" else 24,
                           seconds=12 if task=="cone" else episode_seconds(tier))
    s = (run_cone(args,out,settings,live,pool) if task=="cone" else
         run(args,scenario=scene.parent,live=live,pool=pool,quiet=True))
    s.update(scene=str(plant), prediction_scene=str(prediction))
    (out/"summary.json").write_text(json.dumps(s,indent=2))
    with np.load(out/"episode.npz") as e:
        result = evaluate(task,e,s,live.object_mass)
        assert result["strict"] == (s["status"] == "success")
        warm = e["qpos"][:50]
        warm_path = out.parent.parent/"warm.npy"
        if warm_path.exists():
            np.testing.assert_array_equal(np.load(warm_path),warm)
        else:
            np.save(warm_path,warm)
        hand = "gripper_on_"+task+"_N"
        result.update(hand_impulse_Ns=.02*np.linalg.norm(e[hand][50:],axis=1).sum(),
                      contact_s=.02*e["contact_duty"][50:].sum(),ticks=len(e["qpos"])-50)
        if "forecast_qpos" in e:
            q, actual = e["forecast_qpos"][:,4],e["qpos"][54::5]
            def err(x):
                return np.sqrt(np.mean(np.sum(x*x,axis=1)))
            result.update(prefix_xy_mm=1000*err(q[:,-7:-5]-actual[:,-7:-5]),
                          prefix_ee_mm=1000*err(e["forecast_ee_world"][:,4]-e["ee_world"][54::5]),
                          prefix_force_N=err(e["forecast_"+hand][:,4]-e[hand][54::5]))
            if task == "cone":
                dots = abs((q[:,-4:]*actual[:,-4:]).sum(1))
                result["prefix_angle_deg"] = np.sqrt(np.mean(np.degrees(2*np.arccos(np.clip(dots,0,1)))**2))
        if not params or informed:
            assert result.get("prefix_xy_mm",0.) == result.get("prefix_force_N",0.) == 0
    return dict(**result,task=task,case=case,tier=tier,cem=seed,arm=arm,informed=informed,
                status=s["status"],calls=sum(pool.completed),scheduled=sum(pool.scheduled),
                prediction_sha=digest(prediction),execution_sha=digest(plant),**s["planning"])


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    manifest = dict(cases=CASES, plants=PLANTS, informed_subset=[5501,5511],
                    affinity=sorted(os.sched_getaffinity(0)), python=sys.version,
                    hashes={str(p):digest(p) for p in [POLICY, __file__, unifp_native.__file__,
                            "native/world.cpp","native/rollouts.h","docs/final_lock.json"]})
    assert manifest["affinity"] == list(range(8))
    manifest_text = json.dumps(manifest,indent=2)
    lock = ROOT/"protocol.json"
    if lock.exists():
        assert lock.read_text() == manifest_text
    else:
        lock.write_text(manifest_text)
    rows = []
    for split,task,case,tier,seed in CASES:
        folder = ROOT/task/str(case)
        scene = folder/"nominal/scene.mjb"
        scene.parent.mkdir(parents=True,exist_ok=True)
        if not scene.exists():
            s = (generate_cone(scene.parent,case,cone_model(),tier) if task == "cone" else
                 generate(scene.parent,case,difficulty=tier,box_mass=1.))
            (scene.parent/"scenario.json").write_text(json.dumps(s,indent=2))
        settings = json.loads((scene.parent/"scenario.json").read_text())
        for name, params in PLANTS.items():
            plant = scene if not params else folder/f"{name}.mjb"
            # Correct-plant twins on one prespecified validation scene per task.
            for informed in ([False,True] if params and case in (5501,5511) else [False]):
                for arm in ("A" if split=="dev" else "ABC"):
                    out = folder/name/("informed" if informed else "nominal_prediction")/arm
                    done = out/"result.json"
                    if not done.exists():
                        result = trial(task,tier,case,seed,settings,scene,plant,params,informed,arm,out)
                        result.update(split=split,plant=name)
                        done.write_text(json.dumps(result,indent=2))
                    rows.append(json.loads(done.read_text()))
                    print(split, task, case,name,informed,arm,rows[-1]["status"],flush=True)
    write("docs/supplement_mismatch.csv",rows)


if __name__ == "__main__":
    main()
