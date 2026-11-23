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
        const bool cone = mj_name2id(m, mjOBJ_BODY, "traffic_cone") >= 0;
        if (cone) {
            touch = field<bool>("gripper_touch");
            clearance = field<double>("chassis_cone_gap_m");
        }
        const bool wall = mj_name2id(m, mjOBJ_BODY, "wall") >= 0;
        if (wall) wall_normal = field<double>("wall_normal_N", substeps);
        force = field<double>(wall ? "gripper_on_wall_N" : cone ? "gripper_on_cone_N" : "gripper_on_box_N", 3);
        ground = field<double>(wall ? "ground_on_wall_N" : cone ? "ground_on_cone_N" : "ground_on_box_N", 3);
        peak = field<double>("peak_forward_N");
        duty = field<double>("contact_duty");
        other = field<double>(wall ? "other_robot_wall_impulse_Ns" : cone ? "other_robot_cone_impulse_Ns" : "other_robot_box_impulse_Ns");
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
        if (wall_normal) {
            std::copy_n(w.last_contact.wall_normal.begin(), substeps, wall_normal + row*substeps);
        }
        peak[row] = w.last_contact.peak_forward;
        duty[row] = w.last_contact.contact_duty;
        if (touch) {
            touch[row] = w.last_contact.gripper_touch;
            clearance[row] = w.cone_clearance();
        }
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
    double *wall_normal = nullptr;
    bool *touch = nullptr;
    double *clearance = nullptr;
};

py::dict run(World& world, const Schedules& schedule, bool matched_position) {
    if (schedule.ndim() != 2 || !World::supports_width(schedule.shape(1)) ||
        (matched_position && schedule.shape(1) != 8)) {
        throw std::invalid_argument("expected width 3-6, 8 (required for matched position), or 12");
    }
    const Size ticks = schedule.shape(0);
    const int width = static_cast<int>(schedule.shape(1));
    Trace trace({ticks}, world.plant.model.get());
    for (Size t = 0; t < ticks; ++t) {
        Command command = world.command_at(schedule.data(t, 0), width, matched_position);
        world.step(command);
        trace.record(t, world, command);
    }
    return trace.result;
}

py::dict evaluate(Rollouts& pool, const Snapshot& snapshot, const Schedules& schedules, bool matched_position) {
    if (schedules.ndim() != 3 || schedules.shape(0) < 1 || schedules.shape(1) < 1 ||
        !World::supports_width(schedules.shape(2)) || (matched_position && schedules.shape(2) != 8)) {
        throw std::invalid_argument("expected batches of width 3-6, 8 (required for matched position), or 12");
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
        dispatch = pool.evaluate(snapshot, inputs, count, ticks, width, status.data(), matched_position, record);
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
    py::register_exception<TrajectoryFailure>(m, "TrajectoryFailure", PyExc_RuntimeError);
    py::class_<Snapshot>(m, "Snapshot");
    py::class_<World>(m, "World")
        .def(py::init<const std::string&, const std::string&>())
        .def_property_readonly("base_pose", [](const World& w) {
            const auto* m = w.plant.model.get();
            const int base = mj_name2id(m, mjOBJ_JOINT, "base");
            py::array_t<double> pose(7);
            std::copy_n(w.plant.data->qpos + m->jnt_qposadr[base], 7, pose.mutable_data());
            return pose;
        })
        .def("run", &run, py::arg("schedule"), py::arg("matched_position") = false)
        .def("snapshot", &World::snapshot)
        .def("restore", &World::restore);
    py::class_<Rollouts>(m, "Rollouts")
        .def(py::init<const World&, int>(), py::arg("world"), py::arg("workers") = 1)
        .def("new_episode", &Rollouts::new_episode, py::call_guard<py::gil_scoped_release>())
        .def("evaluate", &evaluate, py::arg("snapshot"), py::arg("schedules"),
             py::arg("matched_position") = false);
}
