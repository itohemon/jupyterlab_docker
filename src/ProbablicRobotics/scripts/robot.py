#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../scripts/")
from ideal_robot import *
from scipy.stats import expon, norm, uniform
matplotlib.rcParams['animation.embed_limit'] = 2**128


# In[2]:


class Robot(IdealRobot):

    def __init__(self, pose, agent=None, sensor=None, color="black", 
                 noise_per_meter=5, # 1[m]あたりの小石の数(5個）
                 noise_std=math.pi/60, # 小石を踏んだときにロボットの向きシータ[deg]に発生する雑音の標準偏差(3[deg])
                 bias_rate_stds=(0.1, 0.1), # バイアスの係数をドローするためのガウス分布の標準偏差(速度、角速度）
                 expected_stuck_time=1e100, # スタックまでの時間の期待値
                 expected_escape_time=1e-100, # スタックから脱出するまでの時間の期待値
                 expected_kidnap_time=1e100, # 誘拐が起こるまでの時間の期待値
                 kidnap_range_x=(-5.0, 5.0), # 誘拐後に置かれるロボットの位置を選ぶ範囲(X座標）
                 kidnap_range_y=(-5.0, 5.0) # 誘拐後に置かれるロボットの位置を選ぶ範囲(Y座標）
                ): 
        super().__init__(pose, agent, sensor, color) # 派生IdealRobotのメソッドを呼び出すときに使う関数super()
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter)) # 指数分布のオブジェクトnoise_pdfを生成する。scale：小石を踏むまでの平均の道のり(1/ラムダ）
        self.distance_until_noise = self.noise_pdf.rvs() # 最初に小石を踏むまでの道のりをセット。rvs()はRandomVariableSの略でランダム変数の生成。
        self.theta_noise = norm(scale=noise_std) # シータに加える雑音を決めるためのガウス分布のオブジェクトを生成
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0]) # loc: 平均
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False
        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0), scale=(rx[1] - rx[0], ry[1] - ry[0], 2*math.pi))

    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu) * time_interval + self.r * abs(omega) * time_interval # 次に小石を踏むまでの道のりdistance_until_noiseを経過時間分だけ減らす
        if self.distance_until_noise <= 0.0: # ゼロ以下なら小石を踏んだ
            self.distance_until_noise += self.noise_pdf.rvs() # 小石を踏んだら次の小石を踏むまでの道のりをセット
            pose[2] += self.theta_noise.rvs() # シータに雑音を加える

        return pose

    def bias(self, nu, omega):
        return nu * self.bias_rate_nu, omega * self.bias_rate_omega
    
    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu * (not self.is_stuck), omega * (not self.is_stuck)

    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose
            
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval) # ノイズを実行する
        self.pose = self.kidnap(self.pose, time_interval)
        if self.sensor: self.sensor.data(self.pose)


# In[3]:


class Camera(IdealCamera):
    def __init__(self, env_map,
                 distance_range=(0.5, 6.0),
                 direction_range=(-math.pi/3, math.pi/3),
                 distance_noise_rate=0.1, # 距離に加える雑音の標準偏差の割合
                 direction_noise=math.pi/90, # 方角に加える雑音の標準偏差
                 distance_bias_rate_stddev=0.1, # 距離に加える一定の距離の標準偏差
                 direction_bias_stddev=math.pi/90, # 方角に加える一定の角度の標準偏差
                 phantom_prob=0.0, # ファントムが出現する確率
                 phantom_range_x=(-5.0, 5.0), # ファントムが出現する座標
                 phantom_range_y=(-5.0, 5.0),  # ファントムが出現する座標
                 oversight_prob=0.1, # 見落とす確率
                 occlusion_prob=0.0, # オクルージョンが起こる確率
                ): 
        super().__init__(env_map, distance_range, direction_range)

        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale=direction_bias_stddev)
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale=(rx[1] - rx[0], ry[1] - ry[0]))
        self.phantom_prob = phantom_prob
        self.oversight_prob = oversight_prob
        self.occlusion_prob = occlusion_prob

    def noise(self, relpos): # relpos:極座標で表現されたセンサ値
        # ガウス分布に従う雑音を混入
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T # 極座標で返す

    def bias(self, relpos):
        return relpos + np.array([relpos[0] * self.distance_bias_rate_std,
                                  self.direction_bias]).T

    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos)
        else:
            return relpos

    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos

    def occlusion(self, relpos):
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0] + uniform.rvs()*(self.distance_range[1] - relpos[0])
            phi = relpos[1]
            return np.array([ell, phi]).T
        else:
            return relpos

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z)
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z) # 観測データにノイズを混入させる
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed            


# In[4]:


if __name__ == '__main__':
    world = World(30, 0.1)
    
    # 地図を生成して３つのランドマークを追加
    m = Map()
    m.append_landmark(Landmark(-4, 2))
    m.append_landmark(Landmark(2, -3))
    m.append_landmark(Landmark(3,  3))
    world.append(m)
    
    # ロボットを作る
    circling = Agent(0.2, 10.0/180*math.pi)
    r = Robot( np.array([0, 0, 0]).T, sensor=Camera(m, occlusion_prob=0.5), agent=circling)
    world.append(r)
    
    # アニメーション実行
    world.draw()

