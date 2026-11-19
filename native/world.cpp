#include "world.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace unifp {

py::dict run(World& world, py::array_t<float, py::array::c_style | py::array::forcecast> commands) {
    if (commands.ndim() != 2 || commands.shape(1) != 12) {
        throw std::invalid_argument("commands must have shape (ticks,12)");
    }
    const py::ssize_t n = commands.shape(0);
    const auto* model = world.plant.model.get();
    const auto* data = world.plant.data.get();
    py::array_t<double> qpos({n, py::ssize_t(model->nq)}), qvel({n, py::ssize_t(model->nv)});
    py::array_t<double> ee({n, py::ssize_t(3)}), force({n, py::ssize_t(3)});
    py::array_t<double> ground({n, py::ssize_t(3)});
    py::array_t<double> peak(n), duty(n), other(n);
    py::array_t<float> actions({n, py::ssize_t(joints)});
    for (py::ssize_t t = 0; t < n; ++t) {
        Command command;
        std::copy_n(commands.data(t, 0), 12, command.begin());
        world.step(command);
        std::copy_n(data->qpos, model->nq, qpos.mutable_data(t, 0));
        std::copy_n(data->qvel, model->nv, qvel.mutable_data(t, 0));
        std::copy_n(data->xpos + 3*world.finger, 3, ee.mutable_data(t, 0));
        std::copy_n(world.action.begin(), joints, actions.mutable_data(t, 0));
        std::copy_n(world.gripper_force.begin(), 3, force.mutable_data(t, 0));
        std::copy_n(world.ground_force.begin(), 3, ground.mutable_data(t, 0));
        peak.mutable_at(t) = world.peak_forward;
        duty.mutable_at(t) = world.contact_duty;
        other.mutable_at(t) = world.other_impulse;
    }
    using namespace pybind11::literals;
    return py::dict("qpos"_a=qpos, "qvel"_a=qvel, "ee_world"_a=ee, "action"_a=actions,
                    "gripper_on_box_N"_a=force, "ground_on_box_N"_a=ground, "peak_forward_N"_a=peak,
                    "contact_duty"_a=duty, "other_robot_box_impulse_Ns"_a=other);
}
}  // namespace unifp

PYBIND11_MODULE(unifp_native, m) {
    py::class_<unifp::World>(m, "World")
        .def(py::init<const std::string&, const std::string&>())
        .def("run", &unifp::run);
}
