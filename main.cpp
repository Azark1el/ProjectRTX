#include <iostream>
#include <Eigen/Dense>
#include <cmath>

class Projectile {
    private:
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
        double mass;
        double drag_coefficient;
        double area;

        static constexpr double g0 = 9.81;
        static constexpr double R = 6371000.0;
        static constexpr double rho0 = 1.225;
        static constexpr double H = 8500.0;

        double computeGravity() const {
            double h = position.z();
            return g0 * std::pow(R / (R + h), 2); 
        }

        double computeAirDensity() const {
            double h = position.z();
            return rho0* std::exp(-h / H);
        }

        Eigen::Vector3d computeDragForce() const {
            double speed = velocity.norm();
            if (speed == 0) return Eigen::Vector3d::Zero();

            double air_density = computeAirDensity();
            return -0.5 * drag_coefficient * air_density * area * speed * velocity.normalized();
        }

    public:
        Projectile(const Eigen::Vector3d& init_position, const Eigen::Vector3d& init_velocity, double init_mass, double init_drag_coefficient, double init_area)
            : position(init_position), velocity(init_velocity), mass(init_mass), drag_coefficient(init_drag_coefficient), area(init_area) {
            acceleration = Eigen::Vector3d::Zero();
        }

        Eigen::Vector3d getPosition() const {
            return position;
        }

        Eigen::Vector3d getVelocity() const {
            return velocity;
        }

        void update(double dt) {
            double g = computeGravity();
            Eigen::Vector3d gravity(0, 0, -g * mass);
            Eigen::Vector3d drag = computeDragForce();
            Eigen::Vector3d total_force = gravity + drag;

            acceleration = total_force / mass;
            velocity += acceleration * dt;
            position += velocity * dt;

            if (position.z() < 0) {
                position.z() = 0;
                velocity.z() = 0;
            }
        }

};

int main() {
    Projectile proj(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(100, 50 ,50), 10, 0.47, 0.05);

    double dt = 0.01;
    double total_time = 1.0;

    for (double t = 0; t < total_time; t += dt) {
        proj.update(dt);
        std::cout << "Time: " << t << " s | Position: " << proj.getPosition().transpose() << " | Velocity: " << proj.getVelocity().transpose() << std::endl;

    }
    return 0;
}