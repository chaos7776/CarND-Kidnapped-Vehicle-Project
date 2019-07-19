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
#define EPS 0.00001

std::default_random_engine gen;

//default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//判断是否已经初始化
	if(is_initialized){
		return;
	}
	//设置粒子数
	num_particles = 100;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
  	normal_distribution<double> dist_y(y, std_y);
  	normal_distribution<double> dist_theta(theta, std_theta);
  	//创建粒子
	for(int i=0; i<num_particles; ++i){
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);		
	}
	//完成初始化，以后不再进行初始化
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	normal_distribution<double> dist_x(0, std_x);
  	normal_distribution<double> dist_y(0, std_y);
  	normal_distribution<double> dist_theta(0, std_theta);

  	for(int i=0; i < num_particles; ++i){
  		double theta = particles[i].theta;
  		if(fabs(yaw_rate) < EPS){
  			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
		}else{
			particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
	      	particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
	      	particles[i].theta += yaw_rate * delta_t;
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
	for(unsigned int i=0; i<observations.size();++i){
		double min_dist = numeric_limits<double>::max();
		int map_id = -1;

		for(unsigned int j=0; j<predicted.size(); ++j)
		{
			double dx = observations[i].x - predicted[j].x;
			double dy = observations[i].y - predicted[j].y;
			double distance = dx*dx + dy*dy;
			if(distance < min_dist){
				min_dist = distance;
				map_id = predicted[j].id;
			}
		}
		observations[i].id = map_id;
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

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	for (int i = 0; i < num_particles; i++) {

    	double x = particles[i].x;
    	double y = particles[i].y;
    	double theta = particles[i].theta;

	    //有效测量

	    vector<LandmarkObs> inRangeLandmarks;
	    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      		float landmarkX = map_landmarks.landmark_list[j].x_f;
      		float landmarkY = map_landmarks.landmark_list[j].y_f;
      		int id = map_landmarks.landmark_list[j].id_i;
      		double dX = x - landmarkX;
      		double dY = y - landmarkY;
      		double distance = sqrt(dX*dX + dY*dY);
      		if ( distance <= sensor_range ) {
        		inRangeLandmarks.push_back(LandmarkObs{ id, landmarkX, landmarkY });
      		}
    	}

	    // 坐标轴转换
	    vector<LandmarkObs> mappedObservations;
	    for(unsigned int j = 0; j < observations.size(); j++) {
	      	double xx = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
	      	double yy = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
	      	mappedObservations.push_back(LandmarkObs{observations[j].id, xx, yy });
	    }

    	// Observation association to landmark.
    	dataAssociation(inRangeLandmarks, mappedObservations);
    	particles[i].weight = 1.0;

    	for(unsigned int j = 0; j < mappedObservations.size(); j++) {
    		double observationX = mappedObservations[j].x;
		    double observationY = mappedObservations[j].y;
		    int landmarkID = mappedObservations[j].id;

		    double landmarkX, landmarkY;

		    //landmarkX = inRangeLandmarks[landmarkID].x;
		    //landmarkX = inRangeLandmarks[landmarkID].y;
		    //这个判断十分有必要，没有误差十分大，为什么？

		    unsigned int k = 0;
		    unsigned int nLandmarks = inRangeLandmarks.size();
		    bool found = false;
		    while( !found && k < nLandmarks ) {
		        if ( inRangeLandmarks[k].id == landmarkID) {
		        	found = true;
		          	landmarkX = inRangeLandmarks[k].x;
		          	landmarkY = inRangeLandmarks[k].y;
		        }
		        k++;
		    }
	      	// 计算权重
	      	double dX = observationX - landmarkX;
	      	double dY = observationY - landmarkY;
	      	//二元高斯
	      	double weight = ( 1/(2*M_PI*std_x*std_y)) * exp(-(dX*dX/(2*std_x*std_y) + (dY*dY/(2*std_x*std_y))));
	      	if (weight < EPS) {
	      		particles[i].weight *= EPS;
	      	} else {
	      		particles[i].weight *= weight;
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
	for(int i=0; i < num_particles; ++i){
		weights.push_back(particles[i].weight);
		if(particles[i].weight > max_weight){
			max_weight = particles[i].weight;
		}
	}
	uniform_real_distribution<double> dist_double(0.0, max_weight);
  	uniform_int_distribution<int> dist_int(0, num_particles - 1);

  	//index.
  	int index = dist_int(gen);
  	double beta = 0.0;
  	vector<Particle> resampled_particles;
  	//构建轮盘进行重采样
  	for(int i=0; i < num_particles; ++i){
  		beta += dist_double(gen);
  		beta += 2.0*max_weight;
  		while(beta > weights[index]){
  			beta -= weights[index];
      		index = (index + 1) % num_particles;
  		}
  		resampled_particles.push_back(particles[index]);
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

    return particle;
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
