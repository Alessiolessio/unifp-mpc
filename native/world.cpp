#include "rollouts.h"

#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace unifp {

using Schedules = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Size = py::ssize_t;

struct Trace {
    Trace(std::vector<Size> shape, const mjModel* m)
        : shape(std::move(shape)), nq(m->nq), nv(m->nv) {
        qpos = field<double>("qpos", nq);
        qvel = field<double>("qvel", nv);
        ee = field<double>("ee_world", 3);
        force = field<double>("gripper_on_box_N", 3);
        ground = field<double>("ground_on_box_N", 3);
        peak = field<double>("peak_forward_N");
        duty = field<double>("contact_duty");
        other = field<double>("other_robot_box_impulse_Ns");
        action = field<float>("action", joints);
        command = field<float>("policy_command", 12);
    }

    void record(Size row, const World& w, const Command& applied) const {
        const auto* d = w.plant.data.get();
        std::copy_n(d->qpos, nq, qpos + row*nq);
        std::copy_n(d->qvel, nv, qvel + row*nv);
        std::copy_n(d->xpos + 3*w.finger, 3, ee + row*3);
        std::copy_n(w.last_contact.gripper_force.begin(), 3, force + row*3);
        std::copy_n(w.last_contact.ground_force.begin(), 3, ground + row*3);
        peak[row] = w.last_contact.peak_forward;
        duty[row] = w.last_contact.contact_duty;
        other[row] = w.last_contact.other_impulse;
        std::copy_n(w.action.begin(), joints, action + row*joints);
        std::copy_n(applied.begin(), 12, command + row*12);
    }

    py::dict result;

private:
    template<class T>
    T* field(const char* name, Size width = 0) {
        auto dims = shape;
        if (width) dims.push_back(width);
        py::array_t<T> array(dims);
        T* data = array.mutable_data();
        // Entire failed suffix stays unscorable, including the first failed tick.
        if (shape.size() == 2) std::fill_n(data, array.size(), std::numeric_limits<T>::quiet_NaN());
        result[name] = array;
        return data;
    }

    std::vector<Size> shape;
    int nq, nv;
    double *qpos, *qvel, *ee, *force, *ground, *peak, *duty, *other;
    float *action, *command;
};

py::dict run(World& world, const Schedules& schedule) {
    if (schedule.ndim() != 2 || !World::supports_width(schedule.shape(1))) {
        throw std::invalid_argument("expected world XYZ with 0-2 body velocities, or 12 policy commands");
    }
    const Size ticks = schedule.shape(0);
    const int width = static_cast<int>(schedule.shape(1));
    Trace trace({ticks}, world.plant.model.get());
    for (Size t = 0; t < ticks; ++t) {
        Command command = world.command_at(schedule.data(t, 0), width);
        world.step(command);
        trace.record(t, world, command);
    }
    return trace.result;
}

py::dict evaluate(Rollouts& pool, const Snapshot& snapshot, const Schedules& schedules) {
    if (schedules.ndim() != 3 || schedules.shape(0) < 1 || schedules.shape(1) < 1 ||
        !World::supports_width(schedules.shape(2))) {
        throw std::invalid_argument("expected batches of world XYZ with 0-2 velocities, or 12 commands");
    }
    const Size count = schedules.shape(0), ticks = schedules.shape(1);
    const int width = static_cast<int>(schedules.shape(2));
    const float* inputs = schedules.data();
    Trace trace({count, ticks}, snapshot.model.get());
    std::vector<RolloutStatus> status(count);
    double dispatch;
    {
        py::gil_scoped_release release;
        auto record = [&](Size row, const World& w, const Command& cmd) {
            trace.record(row, w, cmd);
        };
        dispatch = pool.evaluate(snapshot, inputs, count, ticks, width, status.data(), record);
    }
    py::array_t<bool> valid(count);
    py::array_t<Size> steps(count);
    py::list failures;
    double inference = 0, physics = 0;
    for (Size i = 0; i < count; ++i) {
        const auto& s = status[i];
        valid.mutable_at(i) = s.failure.empty();
        steps.mutable_at(i) = s.steps;
        failures.append(s.failure);
        inference += s.time.inference.count();
        physics += s.time.physics.count();
    }
    trace.result["valid"] = valid;
    trace.result["valid_steps"] = steps;
    trace.result["failure"] = failures;
    trace.result["dispatch_s"] = dispatch;
    trace.result["inference_work_s"] = inference;
    trace.result["physics_work_s"] = physics;
    return trace.result;
}

}  // namespace unifp

PYBIND11_MODULE(unifp_native, m) {
    using namespace unifp;
    py::class_<Snapshot>(m, "Snapshot");
    py::class_<World>(m, "World")
        .def(py::init<const std::string&, const std::string&>())
        .def("run", &run)
        .def("snapshot", &World::snapshot)
        .def("restore", &World::restore);
    py::class_<Rollouts>(m, "Rollouts")
        .def(py::init<const World&, int>(), py::arg("world"), py::arg("workers") = 1)
        .def("evaluate", &evaluate, py::arg("snapshot"), py::arg("schedules"));
}
