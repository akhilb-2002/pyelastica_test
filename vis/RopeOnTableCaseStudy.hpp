//==============================================================================
/*!
//  \file
//  \brief Rope-on-a-table case study
//
//  A Cosserat rod resting on a frictional plane, with one end clamped at the
//  origin and the free end dragged along a prescribed path.  Written against
//  the Elastica++ case-study interface (GazzolaLab/elasticapp).
*/
//==============================================================================

#pragma once
//
#include <elastica/CaseStudy.hpp>
//
#include <elastica/Simulator/Constraints.hpp>
#include <elastica/Simulator/Environments.hpp>
#include <elastica/Simulator/Forcing.hpp>
#include <elastica/Simulator/Simulator.hpp>
#include <elastica/Systems/CosseratRods.hpp>
#include <elastica/Utilities/DefineTypes.h>
#include <elastica/Utilities/Math/Vec3.hpp>
#include <elastica/Utilities/TMPL.hpp>
//
#include <string>
#include <vector>

using String = const char* const;
namespace ec = elastica::configuration;
using CosseratRod = ::elastica::cosserat_rod::CosseratRod;
using ImposedPlane = ::elastica::collision::ImposedPlanePrimitive;

struct RopeOnTableConfiguration {
  static inline auto make_simulator_configuration() {
    auto system_config =
        ec::make_systems_configuration<tmpl::list<CosseratRod>>(
            ec::make_blocking_policy(ec::RestrictSizeAcrossBlockTypes{50UL}),
            ec::make_iteration_policy(
                ec::iterate_across_system_types_in(ec::seq),
                ec::iterate_over_systems_in(ec::seq)));

    using EnvironmentTags =
        tmpl::list<::elastica::environments::tags::GravityField,
                   ::elastica::environments::tags::CollisionSystem>;
    using ImposedPrimitives = tmpl::list<ImposedPlane>;

    auto collision_config =
        ec::make_collision_configuration(ec::seq,   // between each type
                                         ec::seq);  // across each type
    auto gravity_config =
        ec::make_gravity_configuration(::elastica::Vec3{0., 0., 0.});

    auto env_config =
        ec::make_environments_configuration<EnvironmentTags, ImposedPrimitives>(
            ec::seq, gravity_config, collision_config);

    return ec::make_simulator_configuration(
        ec::DefaultDomainConfiguration{}, system_config,
        ec::DefaultParallelConfiguration{},
        ec::DefaultConstraintsConfiguration{}, env_config);
  }

  static inline auto make_simulator() {
    return elastica::make_simulator<::elastica::modules::Constraints,
                                    ::elastica::modules::Environments>(
        make_simulator_configuration());
  }

  static inline auto make_time_stepper() {
    using TimeStepper = ::elastica::timestepper::PositionVerlet;
    return TimeStepper{};
  }
};

struct RopeOnTableCaseStudy
    : public ::elastica::CaseStudyInterface<RopeOnTableCaseStudy> {
  using Configuration = RopeOnTableConfiguration;
  using This = RopeOnTableCaseStudy;
  using ImposedPlane = ::elastica::collision::ImposedPrimitiveAdapter<
      ::elastica::collision::PlanePrimitive>;

  static constexpr String help = {R"help(
Rope on a table
---------------

A rope lies on a frictional plane at z = 0. Node 0 is clamped at the origin;
the free end is dragged kinematically along one of three profiles:

  settle : no drive, the rope simply relaxes under gravity
  sweep  : the tip is dragged through an arc across the table surface
  lift   : the tip is raised off the table and folded back

Intended for producing publication figures via the POVray pipeline.

)help"};

  struct n_elements {
    using type = std::size_t;
    static constexpr String help = {"Number of elements in the rope"};
    static type default_value() { return 40UL; }
  };

  struct length {
    using type = ::elastica::real_t;
    static constexpr String help = {"Length of the rope"};
    static type default_value() { return 1.0; }
  };

  struct radius {
    using type = ::elastica::real_t;
    static constexpr String help = {"Radius of the rope"};
    static type default_value() { return 0.020; }
  };

  struct density {
    using type = ::elastica::real_t;
    static constexpr String help = {"Material density"};
    static type default_value() { return 1000.0; }
  };

  struct youngs_modulus {
    using type = ::elastica::real_t;
    static constexpr String help = {"Youngs modulus of the rope"};
    static type default_value() { return 5e5; }
  };

  struct shear_modulus {
    using type = ::elastica::real_t;
    static constexpr String help = {"Shear modulus of the rope"};
    static type default_value() { return 5e5 / 3.0; }
  };

  struct damping_rate {
    using type = ::elastica::real_t;
    static constexpr String help = {"Internal damping in the rope"};
    static type default_value() { return 1e-4; }
  };

  struct gravitational_acc {
    using type = ::elastica::real_t;
    static constexpr String help = {"Gravitational acceleration along z"};
    static type default_value() { return -9.80665; }
  };

  struct static_friction {
    using type = ::elastica::real_t;
    static constexpr String help = {"Static friction coefficient with table"};
    static type default_value() { return 0.6; }
  };

  struct kinetic_friction {
    using type = ::elastica::real_t;
    static constexpr String help = {"Kinetic friction coefficient with table"};
    static type default_value() { return 0.4; }
  };

  struct contact_stiffness {
    using type = ::elastica::real_t;
    static constexpr String help = {"Contact stiffness with the table"};
    static type default_value() { return 1e3; }
  };

  struct contact_damping {
    using type = ::elastica::real_t;
    static constexpr String help = {"Contact damping with the table"};
    static type default_value() { return 1e-4; }
  };

  struct contact_detection_time_interval {
    using type = ::elastica::real_t;
    static constexpr String help = {"Time between contact detection passes"};
    static type default_value() { return 1e-4; }
  };

  struct drive {
    using type = std::string;
    static constexpr String help = {
        "Tip drive profile: settle, sweep, or lift"};
    static type default_value() { return "lift"; }
  };

  struct drive_time {
    using type = ::elastica::real_t;
    static constexpr String help = {
        "Time over which the tip drive is completed"};
    static type default_value() { return 2.55; }
  };

  struct dt {
    using type = ::elastica::real_t;
    static constexpr String help = {"Time step of simulation"};
    static type default_value() {
      return type(0.002) * (length::default_value() /
                            static_cast<::elastica::real_t>(
                                n_elements::default_value()));
    }
  };

  struct final_sim_time {
    using type = ::elastica::real_t;
    static constexpr String help = {"Final simulation time"};
    static type default_value() { return 3.0; }
  };

  using Parameters =
      tmpl::list<n_elements, length, radius, density, youngs_modulus,
                 shear_modulus, damping_rate, gravitational_acc,
                 static_friction, kinetic_friction, contact_stiffness,
                 contact_damping, contact_detection_time_interval, drive,
                 drive_time, dt, final_sim_time>;

  /* Required functions */
  void setup(InputTypes<This> const& input_parameters);
  void step();
  void run();

  RopeOnTableCaseStudy() = default;
  RopeOnTableCaseStudy(RopeOnTableCaseStudy&&) noexcept = default;

  decltype(Configuration::make_time_stepper()) time_stepper =
      Configuration::make_time_stepper();
  decltype(Configuration::make_simulator()) simulator =
      Configuration::make_simulator();

  using RodRef = decltype(simulator->get_invalid_ref<CosseratRod>());
  RodRef rod_ref = simulator->get_invalid_ref<CosseratRod>();

  using ImposedPlaneRef = decltype(simulator->get_invalid_ref<ImposedPlane>());
  ImposedPlaneRef imposed_plane_ref = simulator->get_invalid_ref<ImposedPlane>();

  ::elastica::real_t final_time;

  // Exposes the rod to Python: r.get_position(), r.get_director(),
  // r.get_radius(), ...
  inline auto communicate() { return rod_ref.get(); }
};

ELASTICA_REGISTER_CASE_STUDY(RopeOnTableCaseStudy);
