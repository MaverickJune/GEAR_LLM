import math

def cal_cpu_reward(cpu_utils,cpu_temps,cluster_num, target_util=0.8):
    lambda_value = 0.15
    # for cpu
    cpu_u_max,cpu_u_min = target_util+0.1,target_util-0.1
    cpu_u_g = target_util
    u,v,l = -0.2,0.21,0.1
    temp_thre = 60
    reward_value = 0.0
    # print('cpu',end=': ')
    for cpu_u, cpu_t in zip(cpu_utils, cpu_temps):
        if cpu_t < temp_thre:
            w = 0.2 * math.tanh(temp_thre-cpu_t)
        else:
            w = -2
        reward_value += w   
            
        if cpu_u < cpu_u_min or cpu_u > cpu_u_max:
            d =lambda_value
        else:
            d = u+v*math.exp(-(cpu_u-cpu_u_g)**2 / (l ** 2))
        reward_value += d
    
    return reward_value/cluster_num