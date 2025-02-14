#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Projectile {
    private:
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
        Eigen::Vector3d target;
        double mass;
        double drag_coefficient;
        double area;

        static constexpr double g0 = 9.81;
        static constexpr double R = 6371000.0;
        static constexpr double rho0 = 1.225;
        static constexpr double H = 8500.0;
        static constexpr double gamma = 1.4;
        static constexpr double R_air = 287.05;
        static constexpr double max_turn_rate = 0.1;
        static constexpr double navigation_constant = 3.0;

        double computeGravity() const {
            double h = position.z();
            return g0 * std::pow(R / (R + h), 2); 
        }

        double computeAirDensity() const {
            double h = position.z();
            return (h > 50000.0) ? 0.0 : rho0* std::exp(-h / H);
        }

        double computeSpeedOfSound() const {
            double h = position.z();
            double T = 288.15 - 0.0065 * h;
            return std::sqrt(gamma * R_air * T);
        }

        double computeDynamicDragCoefficient() const {
            double speed = velocity.norm();
            double mach = speed / computeSpeedOfSound();
            if (mach < 0.8) return 0.3;
            if (mach < 1.2) return 0.6;
            if (mach < 5.0) return 0.4;
            return 0.2;
        }

        Eigen::Vector3d computeDragForce() const {
            double speed = velocity.norm();
            if (speed == 0) return Eigen::Vector3d::Zero();

            double air_density = computeAirDensity();
            double drag_coefficient = computeDynamicDragCoefficient();
            return -0.5 * drag_coefficient * air_density * area * speed * velocity.normalized();
        }

        Eigen::Vector3d computeLiftForce() const {
            double speed = velocity.norm();
            double air_density = computeAirDensity();
            double lift_coefficient = 0.5;
            Eigen::Vector3d lift_direction = Eigen::Vector3d(0, 0, 1);
            return 0.5 * lift_coefficient *air_density * area * speed * speed * lift_direction;
        }

        void adjustTrajectory(double dt) {
            Eigen::Vector3d los_vector = (target - position).normalized();
            Eigen::Vector3d velocity_direction = velocity.normalized();

            Eigen::Vector3d turn_vector = los_vector - velocity_direction;
            double turn_magnitude = turn_vector.norm();
            
            if(turn_magnitude > 0) {
                turn_vector.normalize();
                double max_turn_angle = max_turn_rate * dt;
                velocity_direction = velocity_direction + turn_vector * std::min(turn_magnitude, max_turn_angle);
                velocity_direction.normalize();
                velocity = velocity.norm() * velocity_direction;
            }
        }

    public:
        Projectile(const Eigen::Vector3d& init_position, const Eigen::Vector3d& init_velocity, const Eigen::Vector3d& target_position, double init_mass, double init_area)
            : position(init_position), velocity(init_velocity), target(target_position), mass(init_mass), area(init_area) {
            acceleration = Eigen::Vector3d::Zero();
        }

        Eigen::Vector3d getPosition() const {
            return position;
        }

        Eigen::Vector3d getVelocity() const {
            return velocity;
        }

        bool hasReachedTarget() const {
            return (position - target).norm() < 10 || position.z() <= 0;
        }

        void update(double dt) {
            double g = computeGravity();
            Eigen::Vector3d gravity(0, 0, -g * mass);
            Eigen::Vector3d drag = computeDragForce();
            Eigen::Vector3d lift = computeLiftForce();
            Eigen::Vector3d total_force = gravity + drag + lift;
            
            acceleration = total_force / mass;
            velocity += acceleration * dt;
            adjustTrajectory(dt);
            position += velocity * dt;

            if (position.z() < 0) {
                position.z() = 0;
                velocity.z() = 0;
            }
        }

};

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(-5000, 5000);
    std::uniform_real_distribution<double> alt_dist(3000, 10000);
    std::uniform_real_distribution<double> vel_dist(500, 1500);
    std::uniform_real_distribution<double> elevation_dist(20, 70);
    std::uniform_real_distribution<double> azimuth_dist(0, 360);

    double launch_speed = vel_dist(gen);
    double elevation_angle = elevation_dist(gen) * M_PI / 180;
    double azimuth_angle = azimuth_dist(gen) * M_PI / 180;

    Eigen::Vector3d initial_velocity(launch_speed * cos(elevation_angle) * cos(azimuth_angle), launch_speed * cos(elevation_angle) * cos(azimuth_angle), launch_speed * sin(elevation_angle));

    Eigen::Vector3d initial_position(0, 0, 0);

    Eigen::Vector3d target_position(pos_dist(gen), pos_dist(gen), alt_dist(gen));
    double mass = 4000;
    double area = 0.28;

    Projectile proj(initial_position, initial_velocity,target_position, mass, area);

    double dt = 0.001;
    double total_time = 30.0;

    std::ofstream dataRecord ("SAMdata.txt");
    if (dataRecord.is_open()) {
        dataRecord << std::fixed << std::setprecision(4);

        dataRecord << "Time\tPosition X\tPosition Y\tPosition Z\tVelocityX\tVelocityY\tVelocityZ\n";

        for (double t = 0; t < total_time; t += dt) {
            proj.update(dt);

            Eigen::Vector3d pos = proj.getPosition().transpose();
            Eigen::Vector3d vel = proj.getVelocity().transpose();

            dataRecord << t << "\t" << pos.x() << "\t" << pos.y() << "\t" << pos.z() << "\t" << vel.x() << "\t" << vel.y() << "\t" << vel.z() << "\n";    
        }

        dataRecord.close();
    }

    else std::cout << "Couldn't open file";
    
    return 0;
}