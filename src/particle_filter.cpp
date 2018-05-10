/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i=0; i<num_particles; i++) {
		Particle particle;

		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i=0; i<num_particles; i++) {
		if (fabs(yaw_rate) > 0.001) {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta+yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
    }
    else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i<observations.size(); i++) {
		double min_dist = numeric_limits<double>::max();
		int closest_id = -1;

		for (int j=0; j<predicted.size(); j++) {
		  double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

		  if (distance < min_dist) {
		    min_dist = distance;
		    closest_id = predicted[j].id;
		  }
		}
		observations[i].id = closest_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i=0; i<num_particles; i++) {
		vector<LandmarkObs> transformed_observations;
		for (int j=0; j<observations.size(); j++) {
      double trans_obs_x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
			double trans_obs_y = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
      transformed_observations.push_back(LandmarkObs {observations[j].id, trans_obs_x, trans_obs_y});
    }

		vector<LandmarkObs> predicted_landmarks;
    for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
    	double map_land_x = map_landmarks.landmark_list[j].x_f;
			double map_land_y = map_landmarks.landmark_list[j].y_f;
			int map_land_id = map_landmarks.landmark_list[j].id_i;

      if ((fabs((particles[i].x - map_land_x)) <= sensor_range) && (fabs((particles[i].y - map_land_y)) <= sensor_range)) {
        predicted_landmarks.push_back(LandmarkObs {map_land_id, map_land_x, map_land_y});
      }
    }

		dataAssociation(predicted_landmarks, transformed_observations);

		particles[i].weight = 1.0;
		for (int j=0; j<transformed_observations.size(); j++) {
      double trans_obs_x = transformed_observations[j].x;
      double trans_obs_y = transformed_observations[j].y;
      double trans_obs_id = transformed_observations[j].id;

      for (int k=0; k<predicted_landmarks.size(); k++) {
        double pred_land_x = predicted_landmarks[k].x;
        double pred_land_y = predicted_landmarks[k].y;
        double pred_land_id = predicted_landmarks[k].id;

        if (trans_obs_id == pred_land_id) {
          double multi_prob = (1.0/(2.0*M_PI*std_landmark[0]*std_landmark[1])) * exp(-1.0*(pow((trans_obs_x-pred_land_x), 2)/(2.0*std_landmark[0]*std_landmark[0]) + pow((trans_obs_y - pred_land_y), 2)/(2.0*std_landmark[1]*std_landmark[1])));
          particles[i].weight *= multi_prob;
        }
      }
    }
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<double> weights;
	double max_weight = numeric_limits<double>::min();
	for (int i=0; i<particles.size(); i++) {
    weights.push_back(particles[i].weight);
		if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
  }

  uniform_real_distribution<double> random_weight(0.0, max_weight);
  uniform_int_distribution<int> particle_id(0, num_particles-1);

  int id = particle_id(gen);
  double beta = 0.0;

  vector<Particle> resampled_particles;
  for(int i=0; i<num_particles; i++) {
    beta += random_weight(gen) * 2.0;
    while(beta > weights[id]) {
      beta -= weights[id];
      id = (id + 1) % num_particles;
    }
    resampled_particles.push_back(particles[id]);
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
