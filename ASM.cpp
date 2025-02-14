#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class IMMTracker {
    private:
        std::vector<Eigen::MatrixXd> models;
        Eigen::VectorXd model_probabilities;
        Eigen::MatrixXd transition_matrix;
        Eigen::VectorXd state;
        Eigen::MatrixXd covariance;
        
        void updateModelProbabilities(const Eigen::VectorXd& likelihoods) {
            model_probabilities = transition_matrix * model_probabilities;
            model_probabilities = model_probabilities.array() * likelihoods.array();
            model_probabilities /= model_probabilities.sum();
        }

        Eigen::MatrixXd createCVModel() {
            Eigen::MatrixXd CV = Eigen::MatrixXd::Identity(6, 6);
            CV(0, 2) = CV(1, 3) = CV(4, 5) = 0.1;
            return CV;
        }

        Eigen::MatrixXd createCAModel() {
            Eigen::MatrixXd CA = Eigen::MatrixXd::Identity(6, 6);
            CA(0, 2) = CA(1, 3) = CA(2, 4) = CA(3, 5) = 0.1;
            CA(0, 4) = CA(1, 5) = 0.005;
            return CA;
        }
    
    public:
        IMMTracker() {
            models.push_back(createCVModel());
            models.push_back(createCAModel());
    
            model_probabilities = Eigen::VectorXd::Constant(models.size(), 1.0 / models.size());
            transition_matrix = Eigen::MatrixXd::Constant(models.size(), models.size(), 1.0 / models.size());
            state = Eigen::VectorXd::Zero(6);
            covariance = Eigen::MatrixXd::Identity(6, 6);
        }

        Eigen::VectorXd predict() {
            Eigen::VectorXd predicted_state = Eigen::VectorXd::Zero(6);
            for (size_t i = 0; i < models.size(); ++i) {
                if (models[i].cols() == state.size()) {
                    predicted_state += model_probabilities(i) * (models[i] * state);
                }
            }
            return predicted_state;
        }
        
        void update(const Eigen::VectorXd& measurement) {
            Eigen::VectorXd likelihoods = Eigen::VectorXd::Constant(2, 0.5); 
            updateModelProbabilities(likelihoods);
            state = model_probabilities(0) * (models[0] * state) + model_probabilities(1) * (models[1] * state);
        }
};

class Missile {
    private:
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
        Eigen::Vector3d target;
        Eigen::Vector3d prev_los_vector;
        double mass;
        double drag_coefficient;
        double area;
        IMMTracker tracker;

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
            return (h > 50000.0) ? 0.0 : rho0 * std::exp(-h / H);
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
            if (mach < 3.0) return 0.5;
            if (mach < 5.0) return 0.35;
            return 0.25;
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
            if (speed == 0) return Eigen::Vector3d::Zero();

            double air_density = computeAirDensity();

            Eigen::Vector3d los_vector = (target - position).normalized();
            double angle_of_attack = std::acos(velocity.normalized().dot(los_vector));

            double lift_coefficient = 0.5 * angle_of_attack;
            lift_coefficient = std::max(0.0, std::min(lift_coefficient, 1.2));

            Eigen::Vector3d lift_direction = velocity.cross(Eigen::Vector3d(0, 0, 1)).normalized();

            if (lift_direction.hasNaN() || lift_direction.norm() < 1e-3) {
                lift_direction = Eigen::Vector3d(0, 1, 0);
            }

            return 0.5 * lift_coefficient * air_density * area * speed * speed * lift_direction;
        }

        Eigen::Vector3d proportionalNavigation(const Eigen::Vector3d& los_rate) {
            return navigation_constant * velocity.norm() * los_rate;
        }
    public:
        Missile(const Eigen::Vector3d& init_position, const Eigen::Vector3d& init_velocity, double init_mass, double init_area)
            : position(init_position), velocity(init_velocity), mass(init_mass), area(init_area) {
            acceleration = Eigen::Vector3d::Zero();
            prev_los_vector = Eigen::Vector3d::Zero();
        }

        Eigen::Vector3d getPosition() const {
            return position;
        }

        Eigen::Vector3d getVelocity() const {
            return velocity;
        }

        void update(double dt, const Eigen::Vector3d& target_position) {
            tracker.update(target_position);
            Eigen::VectorXd predicted_target = tracker.predict();

            if (predicted_target.size() >= 3) {
                Eigen::Vector3d new_target = Eigen::Vector3d(predicted_target(0), predicted_target(1), predicted_target(2));
                Eigen::Vector3d los_vector = (new_target - position).normalized();
                Eigen::Vector3d los_rate = (los_vector - prev_los_vector) / dt;
                prev_los_vector = los_vector;

                Eigen::Vector3d pn_acceleration = proportionalNavigation(los_rate);
                double g = computeGravity();
                Eigen::Vector3d gravity(0, 0, -g * mass);
                Eigen::Vector3d drag = computeDragForce();
                Eigen::Vector3d lift = computeLiftForce();
                Eigen::Vector3d total_force = gravity + drag + lift + pn_acceleration;
            
                acceleration = total_force / mass;
                velocity += acceleration * dt;
                position += velocity * dt;
            }
        }

        bool hasReachedTarget() const {
            return position.z() <= 0 || (position - target).norm() < 10;
        }
};

class Target {
    private:
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        double speed;
        double time_since_last_turn;
        int direction_toggle;

    public:
    Target::Target(const Eigen::Vector3d& initial_position, double speed) : position(initial_position), speed(speed), time_since_last_turn(0.0), direction_toggle(1) {
        velocity = Eigen::Vector3d(speed, 0, 0);
    }
    
    void Target::update(double dt) {
        time_since_last_turn += dt;
    
        if (time_since_last_turn >= 2.0) {
            double angle = direction_toggle * M_PI / 4.0;
            double new_vx = speed * std::cos(angle);
            double new_vy = speed * std::sin(angle);
    
            velocity = Eigen::Vector3d(new_vx, new_vy, 0);
            direction_toggle *= -1;
            time_since_last_turn = 0.0;
        }
    
        position += velocity * dt;
    }
    
    Eigen::Vector3d Target::getPosition() const {
        return position;
    };
};

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(-5000, 5000);
    std::uniform_real_distribution<double> alt_dist(8000, 12000);
    std::uniform_real_distribution<double> vel_dist(250, 500);
    std::uniform_real_distribution<double> elevation_dist(-45, 10);
    std::uniform_real_distribution<double> azimuth_dist(0, 360);

    double launch_speed = vel_dist(gen);
    double elevation_angle = elevation_dist(gen) * M_PI / 180;
    double azimuth_angle = azimuth_dist(gen) * M_PI / 180;

    Eigen::Vector3d initial_velocity(
        launch_speed * cos(elevation_angle) * cos(azimuth_angle), 
        launch_speed * cos(elevation_angle) * sin(azimuth_angle), 
        launch_speed * sin(elevation_angle)
    );

    Eigen::Vector3d initial_position(0, 0, alt_dist(gen));
    Eigen::Vector3d target_start(pos_dist(gen), pos_dist(gen), 0);

    double mass = 400;
    double area = 0.2;
    double target_speed = 100.0;

    Missile missile(initial_position, initial_velocity, mass, area);
    Target target(target_start, target_speed);

    double dt = 0.001;
    double total_time = 30.0;

    std::ofstream dataRecord ("ASMdata.txt");
    if (dataRecord.is_open()) {
        dataRecord << std::fixed << std::setprecision(4);

        dataRecord << "Time\tMis. Pos X\tMis. Pos Y\tMis. Pos Z\tMis. Vel X\tMis. Vel Y\tMis. Vel Z\n";

        for (double t = 0; t < total_time; t += dt) {
            target.update(dt);
            Eigen::Vector3d target_pos = target.getPosition().transpose();

            missile.update(dt, target_pos);
            Eigen::Vector3d pos = missile.getPosition().transpose();
            Eigen::Vector3d vel = missile.getVelocity().transpose();

            dataRecord << t << "\t" << pos.x() << "\t" << pos.y() << "\t" << pos.z() << "\t"
            << vel.x() << "\t" << vel.y() << "\t" << vel.z() << "\n";    
        }

        dataRecord.close();
    }

    else std::cout << "Couldn't open file";

    return 0;
}