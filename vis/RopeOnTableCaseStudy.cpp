//==============================================================================
/*!
//  \file
//  \brief Rope-on-a-table case study
*/
//==============================================================================

#include "RopeOnTableCaseStudy.hpp"

#include <elastica/ErrorHandling/FloatingPointExceptions.hpp>
#include <elastica/Simulator/Environments/Kinematics.hpp>
#include <elastica/Simulator/Frames.hpp>
#include <elastica/Simulator/Materials.hpp>
#include <elastica/Utilities/Get.hpp>
#include <elastica/Utilities/Math/Rot3.hpp>
#include <elastica/Utilities/Math/Vec3.hpp>
//
#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <utility>

namespace {

  //! Smoothstep in time: 0 -> 1 with zero slope at both ends.
  inline ::elastica::real_t ramp(::elastica::real_t t,
                                 ::elastica::real_t duration) noexcept {
    const auto s = std::clamp(t / duration, 0.0, 1.0);
    return 0.5 - 0.5 * std::cos(M_PI * s);
  }

  inline bool iequals(std::string const& a, std::string const& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](unsigned char x, unsigned char y) {
                        return std::tolower(x) == std::tolower(y);
                      });
  }

}  // namespace

void RopeOnTableCaseStudy::setup(InputTypes<This> const& input_parameters) {
  const ::elastica::real_t base_length{
      ::elastica::get<length>(input_parameters)};
  const ::elastica::real_t base_radius{
      ::elastica::get<radius>(input_parameters)};
  const std::size_t n_elem = ::elastica::get<n_elements>(input_parameters);

  // ---------------------------------------------------------------- material
  // The friction coefficients here are what make the table frictional; the
  // collision system reads them off the material of the contacting pair.
  elastica::MaterialID my_material = elastica::create_material(
      ::elastica::get<density>(input_parameters),
      0.24,  // coefficient of restitution
      ::elastica::get<youngs_modulus>(input_parameters),
      ::elastica::get<shear_modulus>(input_parameters),
      ::elastica::get<contact_stiffness>(input_parameters),
      ::elastica::get<contact_damping>(input_parameters),
      ::elastica::get<contact_damping>(input_parameters),
      ::elastica::get<static_friction>(input_parameters),
      ::elastica::get<kinetic_friction>(input_parameters));

  // -------------------------------------------------------------------- rope
  // Laid along +x, resting exactly on the plane z = 0.
  const ::elastica::Vec3 origin{0.0, 0.0, base_radius};
  const ::elastica::Vec3 rod_direction{1.0, 0.0, 0.0};
  const ::elastica::Vec3 rod_normal{0.0, 0.0, 1.0};

  using StraightInit = ::elastica::cosserat_rod::StraightCosseratRodInitializer;
  StraightInit straight_initializer{
      StraightInit::NElement{n_elem},
      StraightInit::Material{my_material},
      StraightInit::Radius{base_radius},
      StraightInit::Length{base_length},
      StraightInit::Origin{origin},
      StraightInit::Direction{rod_direction},
      StraightInit::Normal{rod_normal},
      StraightInit::ForceDampingRate{
          ::elastica::get<damping_rate>(input_parameters)},
      StraightInit::TorqueDampingRate{
          ::elastica::get<damping_rate>(input_parameters)},
  };

  rod_ref = simulator->emplace_back<CosseratRod>(
      straight_initializer.initialize<CosseratRod>());

  // ------------------------------------------------------- clamp the one end
  namespace eco = ::elastica::constraints;

  const ::elastica::Rot3 clamped_director =
      ::elastica::director(rod_ref.get(), 0UL);

  auto fixed_end_generator = eco::make_constraint(
      eco::make_value_constraint(
          [origin, clamped_director](::elastica::real_t /*time*/,
                                     auto& rod_system, auto index) {
            ::elastica::position(rod_system, index) = origin;
            ::elastica::director(rod_system, index) = clamped_director;
          }),
      eco::make_rate_constraint([](::elastica::real_t /*time*/,
                                   auto& rod_system, std::size_t index) {
        ::elastica::velocity(rod_system, index) = 0.0;
        ::elastica::angular_velocity(rod_system, index) = 0.0;
      }));

  simulator->constrain(rod_ref).at(0UL).with(fixed_end_generator());

  // ------------------------------------------------ drag the free end around
  const std::string profile = ::elastica::get<drive>(input_parameters);
  const ::elastica::real_t T = ::elastica::get<drive_time>(input_parameters);

  if (not iequals(profile, "settle")) {
    const bool is_lift = iequals(profile, "lift");
    const ::elastica::real_t L = base_length;
    const ::elastica::real_t r = base_radius;

    // Prescribed tip position; velocity is its analytic derivative so that the
    // rate constraint stays consistent with the value constraint.
    auto tip_position = [=](::elastica::real_t t) -> ::elastica::Vec3 {
      const auto s = ramp(t, T);
      if (is_lift)
        return ::elastica::Vec3{L * (1.0 - 0.55 * s), 0.12 * L * s,
                                r + 0.55 * L * s};
      const auto ang = (75.0 * M_PI / 180.0) * s;
      const auto rad = L * (1.0 - 0.18 * s);
      return ::elastica::Vec3{rad * std::cos(ang), rad * std::sin(ang), r};
    };

    auto tip_velocity = [=](::elastica::real_t t) -> ::elastica::Vec3 {
      constexpr ::elastica::real_t h = 1e-6;
      const auto forward = tip_position(t + h);
      const auto backward = tip_position(t - h);
      return ::elastica::Vec3{(forward[0] - backward[0]) / (2.0 * h),
                              (forward[1] - backward[1]) / (2.0 * h),
                              (forward[2] - backward[2]) / (2.0 * h)};
    };

    auto tip_generator = eco::make_constraint(
        eco::make_value_constraint([tip_position](::elastica::real_t time_v,
                                                 auto& rod_system, auto index) {
          // constrained at the last element -> last node is index + 1
          ::elastica::position(rod_system, index + 1UL) = tip_position(time_v);
        }),
        eco::make_rate_constraint([tip_velocity](::elastica::real_t time_v,
                                                auto& rod_system,
                                                std::size_t index) {
          ::elastica::velocity(rod_system, index + 1UL) = tip_velocity(time_v);
        }));

    simulator->constrain(rod_ref)
        .at(::elastica::end - 1UL)
        .with(tip_generator());
  }

  // ------------------------------------------------------------- the table
  // Plane at the origin with its normal along +z.
  const ::elastica::Vec3 plane_position{0.0, 0.0, 0.0};
  auto plane_normal = {0.0, 0.0, 1.0};
  auto plane_binormal = {0.0, 1.0, 0.0};
  auto plane_tangent = {1.0, 0.0, 0.0};
  ::elastica::Rot3 Q = {plane_normal, plane_binormal, plane_tangent};
  using Spans = typename ImposedPlane::Spans;

  namespace ek = elastica::kinematics;
  auto always_zero_ = [](::elastica::real_t) noexcept -> ::elastica::real_t {
    return 0.0;
  };
  // The table does not move, but an ImposedPlane still needs a kinematics
  // buffer, so every axis gets a zero position/velocity profile.
  auto translations_buffer = ek::make_translations(
      ek::translate_along<elastica::EulerianFrame::X>(
          ek::mark_position(always_zero_), ek::mark_velocity(always_zero_)),
      ek::translate_along<elastica::EulerianFrame::Y>(
          ek::mark_position(always_zero_), ek::mark_velocity(always_zero_)),
      ek::translate_along<elastica::EulerianFrame::Z>(
          ek::mark_position(always_zero_), ek::mark_velocity(always_zero_)));

  imposed_plane_ref = simulator->emplace_back<ImposedPlane>(
      ::elastica::collision::PlanePrimitive(0UL, 0UL, true, plane_position, Q,
                                            Spans{1e4, 1e4}, my_material),
      std::move(translations_buffer));

  auto& collision_solver =
      simulator->get_environment<elastica::environments::tags::CollisionSystem>();
  collision_solver->time_interval_between_detection() =
      ::elastica::get<contact_detection_time_interval>(input_parameters);

  // ------------------------------------------------------------------ gravity
  const ::elastica::Vec3 gravity_acc_vector{
      0.0, 0.0, ::elastica::get<gravitational_acc>(input_parameters)};
  auto& gravity_field =
      simulator->get_environment<elastica::environments::tags::GravityField>();
  gravity_field->gravity() = gravity_acc_vector;

  simulator->finalize();

  final_time = ::elastica::get<final_sim_time>(input_parameters);
  set_dt(::elastica::get<dt>(input_parameters));
}

void RopeOnTableCaseStudy::step() {
  simulator->step(time_stepper, get_current_time(), get_dt(),
                  get_current_iteration());
}

void RopeOnTableCaseStudy::run() {
  /* NOTE : this won't get called from the python side */
  enable_floating_point_exceptions();

  const std::size_t n_steps = (final_time / current_dt());
  for (std::size_t i = 0UL; i < n_steps; ++i) {
    step();
  }
}
