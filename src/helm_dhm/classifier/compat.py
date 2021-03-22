'''
HELM classifier support function library
'''
import glob
import json
import os

import pickle
import shutil
import logging
import itertools
import datetime
import time
import yaml
import copy

import matplotlib.pyplot as pyplot
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib        as plt
import pylab             as pl
import os.path           as op

from pathlib                   import Path
from scipy.stats               import multivariate_normal
from tqdm                      import tqdm
from random                    import seed as rseed
from sklearn.ensemble          import RandomForestClassifier
from scipy.stats               import norm, chi2
from typing                    import NamedTuple
from scipy                     import interp
from sklearn.metrics           import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection   import StratifiedKFold

from utils.dir_helper          import get_batch_subdir, get_exp_subdir

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.grid'] = False

class NormallyDistributed(object):
    """
    Stores mu and covarance, and specifies how to print values.
    """

    EPSILON = 1e-9

    def __init__(self, mu, cov):
        if np.asarray(cov).size != np.asarray(mu).size ** 2:
            raise ValueError('Covariance matrix must have the dimensions of mu squared.')
        if not (np.asarray(cov) == np.transpose(np.asarray(cov))).all():
            raise ValueError('Covariance matrix must equal its transpose.')
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)

    def store(self):
        return {
            'mu': list(self.mu.flat),
            'cov': [list(row.flat) for row in self.cov]
        }

    def __repr__(self):
        return ('%s(mu=[%f, %f], cov=[[%f, %f], [%f, %f]])'
                % ((self.__class__.__name__, self.mu[0], self.mu[1]) + tuple(self.cov.flat)))


class Particle(NormallyDistributed):

    def __init__(self, mu, cov):
        super().__init__(mu=mu, cov=cov)

    @staticmethod
    def aggregate(particles):
        """
        Returns a single particle with aggregated mu and cov of particles passed in.
        """
        if len(particles) == 0:
            return None
        if not all(isinstance(x, NormallyDistributed) for x in particles):
            raise Exception('Particles must be NormallyDistributed')
        mu = np.average(np.vstack(
            [p.mu for p in particles]
        ), axis=0)
        cov = np.average(np.dstack(
            [p.cov for p in particles]
        ), axis=2)

        return Particle(mu, cov)


class Projectile(NormallyDistributed):

    def __init__(self, mu, cov):
        super().__init__(mu=mu, cov=cov)

    def project(self, rate, delta_t):
        """
        Updates mu and cov. Returns object type definition was called on
        """
        if not isinstance(rate, NormallyDistributed):
            raise Exception('When projecting, rate must be NormallyDistributed')
        if delta_t <= 0:
            raise ValueError('Timestep must be positive')

        mu_new = self.mu + delta_t * rate.mu
        cov_new = self.cov + (delta_t ** 2) * rate.cov

        return self.__class__(mu_new, cov_new)


class Position(Projectile):
    """
    Position class stores mu and covariance. Is normally distributed and has updates defined in Projectile class
    """

    def __init__(self, mu, cov):
        super().__init__(mu, cov)
        self.inv_cov = None

    @staticmethod
    def from_particle(particle):
        return Position(particle.mu, particle.cov)

    def dist(self, mu):
        """
        Computes Mahalanobis distance from mu to mean position
        """
        if self.inv_cov is None:
            # Add (epsilon * I) for numerical stability
            self.inv_cov = np.linalg.inv(self.cov + self.EPSILON * np.eye(2))
        if not (np.array(mu).shape == np.array(self.mu).shape):
            raise ValueError('Mu should have same shape as Position mu')
        diff = self.mu - mu
        return float(np.sqrt(np.dot(diff, np.dot(self.inv_cov, diff))))


class Velocity(Projectile):
    """
    Velocity object has mu and cov, and can be projected to next location
    """
    def __init__(self, mu, cov):
        super().__init__(mu=mu, cov=cov)


class Acceleration(NormallyDistributed):
    """
    Acceleration object has mu and cov
    """
    def __init__(self, mu, cov):
        """

        Parameters
        ----------
        mu
        cov
        """
        super().__init__(mu=mu, cov=cov)


def sigma_to_mahalanobis(sigma):
    """
    Returns the Mahalobis distance for a 2-d Gaussian that covers the
    same probability mass as the given sigma in a 1-d Gaussian
    """
    if sigma < 0:
        raise ValueError('Sigma cannot be negative')

    percentile = 1.0 - 2 * (1.0 - norm.cdf(sigma))

    return chi2.ppf(percentile, 2)

def uncertainty_major_radius(cov, sigma=3.0):
    """
    Returns uncertainty of major radius
    """

    if cov.shape[0] != cov.shape[1]:
        raise ValueError('Covariance must be square matrix')

    factor = sigma_to_mahalanobis(sigma)
    vals, _ = np.linalg.eig(cov)
    return np.max(2 * np.sqrt(factor * vals))

class BaseFeatures(object):

    def __init__(self, **data):
        self.track_length = data.get('track_length')
        self.max_velocity = data.get('max_velocity')
        self.mean_velocity = data.get('mean_velocity')
        self.stdev_velocity = data.get('stdev_velocity')
        self.autoCorr_vel_lag1 = data.get('autoCorr_vel_lag1')
        self.autoCorr_vel_lag2 = data.get('autoCorr_vel_lag2')
        self.max_stepAngle = data.get('max_stepAngle')
        self.mean_stepAngle = data.get('mean_stepAngle')
        self.autoCorr_stepAngle_lag1 = data.get('autoCorr_stepAngle_lag1')
        self.autoCorr_stepAngle_lag2 = data.get('autoCorr_stepAngle_lag2')
        self.max_accel = data.get('max_accel')
        self.mean_accel = data.get('mean_accel')
        self.stdev_accel = data.get('stdev_accel')
        self.autoCorr_accel_lag1 = data.get('autoCorr_accel_lag1')
        self.autoCorr_accel_lag2 = data.get('autoCorr_accel_lag2')
        self.ud_x = data.get('ud_x')
        self.ud_y = data.get('ud_y')
        self.theta_displacement = data.get('theta_displacement')
        self.rel_vel = data.get('rel_vel')
        self.rel_theta_displacement = data.get('rel_theta_displacement')
        self.rel_dir_dot = data.get('rel_dir_dot')

    def get_active_features(self):
        out_dict = {}

        # Don't rely on self.__dict__ or vars(self) because other attributes may get included
        # instead define the dictionary of class attributes expected directly
        # if you know a good work around feel free to add it
        track_features = {
            'track_length': self.track_length,
            'max_velocity': self.max_velocity,
            'mean_velocity': self.mean_velocity,
            'stdev_velocity': self.stdev_velocity,
            'autoCorr_vel_lag1': self.autoCorr_vel_lag1,
            'autoCorr_vel_lag2': self.autoCorr_vel_lag2,
            'max_stepAngle': self.max_stepAngle,
            'mean_stepAngle': self.mean_stepAngle,
            'autoCorr_stepAngle_lag1': self.autoCorr_stepAngle_lag1,
            'autoCorr_stepAngle_lag2': self.autoCorr_stepAngle_lag2,
            'max_accel': self.max_accel,
            'mean_accel': self.mean_accel,
            'stdev_accel': self.stdev_accel,
            'autoCorr_accel_lag1': self.autoCorr_accel_lag1,
            'autoCorr_accel_lag2': self.autoCorr_accel_lag2,
            'ud_x': self.ud_x,
            'ud_y': self.ud_y,
            'theta_displacement': self.theta_displacement,
            'rel_vel': self.rel_vel,
            'rel_theta_displacement': self.rel_theta_displacement,
            'rel_dir_dot': self.rel_dir_dot
        }

        for key in track_features.keys():
            result = track_features[key]
            if result is not None:
                out_dict[key] = result
        return out_dict


class ParticleTrack(BaseFeatures):
    """
        Particle Track object
    """
    NEXT_ID = 0

    def __init__(self, particle, *args, **kwargs):

        super().__init__(**kwargs)
        if isinstance(particle, Particle):
            t0, mu_v0, cov_v0, mu_a0, cov_a0 = args
            self.particles = [particle]
            self.times = [t0]
            self.positions = [Position.from_particle(particle)]
            self.velocities = [Velocity(mu_v0, cov_v0)]
            self.acceleration = Acceleration(mu_a0, cov_a0)
            self.average_velocity = None
            self.track_id = None

        else:
            times, positions, velocities, acceleration, averages, tid, classification, prob_mot = args
            self.particles = particle
            self.times = times
            self.positions = positions
            self.velocities = velocities
            self.acceleration = acceleration
            self.average_velocity = averages
            self.track_id = tid
            self.classification = classification
            self.prob_mot = prob_mot

    @classmethod
    def next_track_id(cls):
        try:
            return cls.NEXT_ID
        finally:
            cls.NEXT_ID += 1

    @staticmethod
    def load(trackfile):
        with open(trackfile, 'r') as f:
            data = json.load(f)
        particles = [(None if p is None else Particle(**p))
                     for p in data['particles']]
        times = [t for t in data['times']]
        positions = [Position(**p) for p in data['positions']]
        velocities = [Velocity(**v) for v in data['velocities']]
        acceleration = Acceleration(**data['acceleration'])
        avg_vel = [np.array(av) for av in data['average_velocity']]
        tid = data['track_id']
        try:
            classification = data["classification"]
            prob_mot = data["probability_motility"]
        except:
            classification = None
            prob_mot = None

        return ParticleTrack(particles, times, positions,
                             velocities, acceleration, avg_vel, tid, classification, prob_mot, **data)

    @staticmethod
    def load_from_dict(data: dict):

        particles = [(None if p is None else Particle(**p))
                     for p in data['particles']]
        times = [t for t in data['times']]
        positions = [Position(**p) for p in data['positions']]
        velocities = [Velocity(**v) for v in data['velocities']]
        acceleration = Acceleration(**data['acceleration'])
        avg_vel = [np.array(av) for av in data['average_velocity']]
        tid = data['Track_ID']
        try:
            classification = data["classification"]
            prob_mot = data["probability_motility"]
        except:
            classification = None
            prob_mot = None

        return ParticleTrack(particles, times, positions,
                             velocities, acceleration, avg_vel, tid, classification, prob_mot, **data)

    def store(self):
        """
        Store the ParticleTrack objects
        """
        avg_vel = None
        if self.average_velocity is not None:
            avg_vel = [list(av.flat) for av in self.average_velocity]
        return {
            'particles': [(p.store() if p is not None else None)
                          for p in self.particles],
            'times': [t for t in self.times],
            'positions': [p.store() for p in self.positions],
            'velocities': [v.store() for v in self.velocities],
            'acceleration': self.acceleration.store(),
            'average_velocity': avg_vel,
            'track_id': self.track_id,
        }

    def save(self, trackfile):
        data = self.store()
        with open(trackfile, 'w+') as f:
            json.dump(data, f, indent=2)

    def __len__(self):
        """
        This definition will count "None" particles
        """
        return len(self.particles)

    def n_obs(self):
        """
        Counts the actual particles
        """
        return len([p for p in self.particles
                    if p is not None])

    def _trim(self):
        """
        Remove final trailing particles
        """
        while len(self.particles) > 0 and self.particles[-1] is None:
            self.particles.pop(-1)
            self.times.pop(-1)
            self.positions.pop(-1)
            self.velocities.pop(-1)

    def _record_avg_velocities(self, velocity_dict):
        self.average_velocity = np.array([velocity_dict[t] for t in self.times])

    def finish(self, velocity_dict):
        self._trim()
        self._record_avg_velocities(velocity_dict)
        self.track_id = ParticleTrack.next_track_id()

    def position_uncertainty(self, sigma=3.0):
        return uncertainty_major_radius(self.positions[-1].cov, sigma)

    def project(self, delta_t=1.0):
        return self.positions[-1].project(self.velocities[-1], delta_t)

    def update(self, t, particle):
        """
        Update the track. If no new particle detect, propagate with prior velocity
        """
        delta_t = t - self.times[-1]
        if particle is None:
            p_new = self.positions[-1].project(self.velocities[-1], delta_t)
            v_new = self.velocities[-1].project(self.acceleration, delta_t)
        else:
            p_new = Position.from_particle(particle)
            p_last = self.positions[-1]
            v_mu = (p_new.mu - p_last.mu) / delta_t
            # Covariance of a difference of Gaussians is sum of covariances
            # Covariance scales with the square of the scale of the Gaussian
            v_cov = (p_new.cov + p_last.cov) / (delta_t ** 2)
            v_new = Velocity(v_mu, v_cov)

        self.particles.append(particle)
        self.times.append(t)
        self.positions.append(p_new)
        self.velocities.append(v_new)


def load_tracks(track_directory):
    """
        Loads track JSON files
    """
    trackfiles = sorted(glob.glob(os.path.join(track_directory, '*.track')))
    return [ParticleTrack.load(t) for t in trackfiles]
