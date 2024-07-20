import osmnx as ox
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import pandas as pd

# Parameters ########################################################
alpha = 0.4
my_lambda = 0.5

K = 8  # maximal number of chargers at a station
P = 3 # number of periods
H = 60 # period length (min)
RADIUS_MAX = 2.5  # [radius_max] = m
INSTALL_FEE = np.array([300, 750, 28000])  # fee per installing a charger of type 1, 2 or 3. [fee] = $
CHARGING_POWER = np.array([7, 22, 50])  # [power] = kW, rounded
CHARGING_POWER_MP = 42 # KW,rounded
BATTERY_MP = 70 # KW,rounded
MP_BATCH = 10 # number of each batch of MPs
M = 30 # number of batches MP

BATTERY = 70  # battery capacity, [BATTERY] = kWh

BUDGET = 5 * 10 ** 6  # [B] = €
price_parkingplace = 200 * 3.5 * 2  # in €

time_unit = 1  # [time_unit] = h, introduced for getting the units correctly
capacity_unit = 1  # [cap_unit] = kW, introduced for getting the units correctly
VELOCITY = 23  # based on km per hour, but here dimensionless

my_inf = 10 ** 6
my_dis_inf = 10 ** 7


def prepare_nodes_inf(file_path):
    "准备节点信息"
    inf = pd.read_csv(file_path + '/information.csv', index_col=None, header=0)
    occ = pd.read_csv(file_path + '/occupancy.csv', index_col=0, header=0)
    # create test data (last day)
    cd = occ.tail(288)
    demand = pd.DataFrame(index=range(1, 25), columns=occ.columns)
    for i in range(24):
        demand.loc[i + 1] = cd.iloc[i * 12:(i + 1) * 12].max().values
    demand_dict = demand.loc[1].to_dict()
    # demand = np.array(demand)
    my_node_list = []
    for i in range(len(inf)):
        grid = int(inf['grid'][i]) # node id
        dict = {'x': inf['la'][i],
                'y': inf['lon'][i],
                'demand': demand[str(grid)][0:P].values, # 现在及未来需求，以1小时为unit，现包含3个units
                'charging station': [None] * P, # 被分配的充电站,
                # 'cs':[],
                'distance': [None] * P, # 前往充电站的行驶距离
                'private CS': 0,
                }
        my_node_list.append((grid, dict))
    return my_node_list

def social_efficiency_upper_bound(my_node, my_node_list):
    """
    calculate the social efficiency for each node
    """

    I1_max = 0  # dimensionless
    for other_node in my_node_list:
        # calculate distance with haversine approximation
        if haversine(my_node, other_node) <= RADIUS_MAX:
            I1_max += 1
    my_node[1]["I1_max"] = I1_max
    delta_benefit = I1_max
    delta_benefit /= 100  # does not matter as we scale here
    upper_bound = my_lambda * delta_benefit
    return upper_bound

def haversine(s_pos, my_node):
    """
    yields the approximate distance of two GPS points, middle computational cost
    """
    lon1, lat1 = s_pos[1]['x'], s_pos[1]['y']
    R_earth = 6372.800  # approximate radius of earth. [R_earth] = m
    lon2, lat2 = my_node[1]['x'], my_node[1]['y']
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R_earth * c  # [distance] = m
    if distance < 0.0001:  # to avoid ZeroDivisionError
        distance = 0.0001
    return distance

def installment_fee(my_station):
    """
    returns cost to install the respective chargers at that position
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    charger_cost = np.sum(INSTALL_FEE * s_x)
    fee = charger_cost
    s_dict["fee"] = fee  # [fee] = €
    return my_station

def charging_capability(my_station, mp_list):
    """
    returns the summed up charging capability of the CS
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = np.sum(CHARGING_POWER * s_x)
    s_dict["fixed capability"] = total_capacity  # [capability] = kw
    sum_capability = [total_capacity] * P  # [capability] = kw
    # following step is to add the capability provided by MPs
    for m in range(len(mp_list)):
        for p in range(P):
            if mp_list[m].mp[0][p] == s_pos[0]:
                remaining_power = mp_list[m].mp[1][p]
                arrival_time = mp_list[m].mp[2][p]
                accessible_time_ratio = 1 - min((arrival_time - p*H)/H, 1)
                if mp_list[m].mp[1][p] > CHARGING_POWER_MP: # if the remaining electricity is larger than Power, this MP can be used.
                    sum_capability[p] += CHARGING_POWER_MP * accessible_time_ratio * MP_BATCH
    s_dict["capability"] = sum_capability
    return my_station

def influence_radius(my_station):
    """
    gives the radius of the nodes whose charging demand the CS could satisfy
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = s_dict["capability"]
    radius_s = []
    for p in range(P):
        radius_s.append(RADIUS_MAX * 1 / (1 + np.exp(-total_capacity[p] / (100 * capacity_unit))))
    s_dict["radius"] = radius_s  # [radius] = m
    return my_station

def get_demand(my_node):
    return my_node[1]["demand"]



def cost_single(my_node, my_station, my_node_dict, my_cost_dict, p):
    """
    calculate the social cost for one station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    # check if distance has to be calculated
    if s_pos[0] in my_node_dict[my_node[0]]:
        distance = my_node_dict[my_node[0]][s_pos[0]]
    else:
        distance = haversine(s_pos, my_node)
        my_node_dict[my_node[0]][s_pos[0]] = distance

    if s_pos[0] in my_node_dict[my_node[0]]:
        distance = my_node_dict[my_node[0]][s_pos[0]]
    else:
        distance = haversine(s_pos, my_node)
        my_node_dict[my_node[0]][s_pos[0]] = distance
    # check if cost has to be calculated
    try:
        _a = my_cost_dict[p][my_node[0]]
    except KeyError:
        my_cost_dict[p][my_node[0]] = {}
    if s_pos[0] in my_cost_dict[p][my_node[0]]:
        cost_node = my_cost_dict[p][my_node[0]][s_pos[0]]
    else:
        cost_travel = alpha * distance / VELOCITY
        cost_boring = (1 - alpha) / distance * (s_dict["W_s"][p] + 1 / s_dict["service rate"][p])
        cost_node = get_demand(my_node)[p] * (cost_travel + cost_boring)  # 目前是把所有周期的demand累加算成本，未来要修改
        my_cost_dict[p][my_node[0]][s_pos[0]] = cost_node
    return cost_node, my_node_dict, my_cost_dict

def station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict):
    """
    output station assignment: Each node gets assigned the charging station with minimal social cost
    """
    for p in range(P):
        for the_node in my_node_list:
            cost_list = [cost_single(the_node, my_station, my_node_dict, my_cost_dict, p) for my_station in my_plan]
            costminindex = cost_list.index(min(cost_list))
            chosen_station = my_plan[costminindex]
            s_pos = chosen_station[0]
            the_node[1]["charging station"][p] = s_pos[0]
            the_node[1]["distance"][p] = my_node_dict[the_node[0]][s_pos[0]]
    return my_node_list, my_node_dict, my_cost_dict

def total_number_EVs(my_station, my_node_list):
    """
    yields total number of EVs coming to S in a unit time interval for charging
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    s_dict["D_s"] = [None] * P
    for p in range(P):
        D_s = sum([1 / my_node[1]["distance"][p] * get_demand(my_node)[p] if my_node[1]["charging station"][p] == s_pos[0]
                   else 0 for my_node in my_node_list]) # demand和，将来修改
        s_dict["D_s"][p] = D_s  # dimensionless
    return my_station


def service_rate(my_station):
    """
    returns how many cars can be served within one hour
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    s_dict["service rate"] = [None] * P
    for p in range(P):
        s_dict["service rate"][p] = s_dict["capability"][p] / BATTERY  # [service rate] = 1/h
    return my_station


def W_s(my_station):
    """
    returns the expected value of waiting time
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    s_dict["W_s"] = [None] * P
    for p in range(P):
        tau_s = 1 / s_dict["service rate"][p]
        rho_s = s_dict["D_s"][p] * tau_s * time_unit  # dimensionless (shortened away)
        if rho_s >= 1:
            my_W_s = my_inf
            s_dict["W_s"][p] = my_W_s
        else:
            my_W_s = rho_s * tau_s / (2 * (1 - rho_s))  # W_s = expected waiting time at S, [W_s] = h
            s_dict["W_s"][p] = my_W_s
    return my_station


def s_dictionnary(my_station, my_node_list, my_mps_list):
    """
    returns the dictionnary for the station
    """
    my_station = installment_fee(my_station)
    my_station = charging_capability(my_station, my_mps_list)
    my_station = influence_radius(my_station)
    my_station = total_number_EVs(my_station, my_node_list)
    my_station = service_rate(my_station)
    my_station = W_s(my_station)
    return my_station


def node_coverage(my_plan, my_node):
    """
    yields the number of nodes within the influence radius of the station
    """
    I_3 = 0
    for p in range(P):
        I_1, I_2 = 0, 0
        for my_station in my_plan:
            s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
            radius_s = s_dict["radius"][p]
            distance = haversine(s_pos, my_node)
            if distance <= radius_s:
                I_1 += 1
        for ith in range(I_1):
            I_2 += 1 / (ith + 1)
        I_3 += I_2
    single_benefit = I_3/P
    return single_benefit

def Coverage(my_plan, my_node_list):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    my_benefit = 0
    for my_node in my_node_list:
        I3 = node_coverage(my_plan, my_node)
        my_benefit += I3
    my_benefit /= len(my_node_list)
    return my_benefit

def travel_cost(my_node_list):
    """ yields the estimated travel time of all vehicles
     """
    my_cost_travel = 0
    for p in range(P):
        my_cost_travel += sum(sum([my_node[1]["distance"][p] * get_demand(my_node) / VELOCITY for my_node in my_node_list]))
    return my_cost_travel

def charging_time(my_plan):
    """
    yields the total charging time given the capability of the CS of the charging plan
    """
    my_charg_time = 0
    for p in range(P):
        my_charg_time += sum([my_station[2]["D_s"][p] / my_station[2]["service rate"][p] for my_station in my_plan])
    return my_charg_time / time_unit


def waiting_time(my_plan):
    """
    returns the average total waiting time of the charging plan
    """
    my_wait_time = 0
    for p in range(P):
        my_wait_time += sum([my_station[2]["D_s"][p] * my_station[2]["W_s"][p] for my_station in my_plan])
    return my_wait_time / time_unit

def Cost(my_plan, my_node_list):
    """
    returns the social cost, i.e. the negative side of the charging plan
    """
    cost_travel = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_plan)  # dimensionless
    wait_time = waiting_time(my_plan)  # dimensionless
    cost_boring = charg_time + wait_time  # dimensionless
    my_social_cost = alpha * cost_travel + (1 - alpha) * cost_boring
    return my_social_cost

def existing_score(my_existing_plan, my_node_list):
    """
    computes the score of the existing infrastructure
    """
    my_benefit = Coverage(my_existing_plan, my_node_list)
    travel_time = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_existing_plan)  # dimensionless
    wait_time = waiting_time(my_existing_plan)
    cost_boring = charg_time + wait_time  # dimensionless
    my_cost = alpha * travel_time + (1 - alpha) * cost_boring
    return my_benefit, my_cost, charg_time, wait_time, travel_time


def norm_score(my_plan, my_node_list, norm_benefit, norm_charg, norm_wait, norm_travel):
    """
    same as score, but normalised.
    """
    my_score = -my_inf
    if not my_plan:
        return my_score
    benefit = Coverage(my_plan, my_node_list) / norm_benefit
    cost_travel = travel_cost(my_node_list) / norm_travel  # dimensionless
    charg_time = charging_time(my_plan) / norm_charg  # dimensionless
    wait_time = waiting_time(my_plan) / norm_wait  # dimensionless
    cost = (alpha * cost_travel + (1 - alpha) * (charg_time + wait_time)) / 3
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost, charg_time, wait_time, cost_travel


def score(my_plan, my_node_list):
    """
    returns the final result, i.e., the social score
    """
    my_score = -my_inf
    benefit = 0
    cost = 0
    if not my_plan:
        return my_score, benefit, cost
    benefit = Coverage(my_plan, my_node_list)  # dimensionless
    cost = Cost(my_plan, my_node_list)
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost


# Constraints checks ############################################################################
def station_capacity_check(my_plan):
    """
    check if number of stations exceed capacity
    """
    for my_station in my_plan:
        s_x = my_station[1]
        if sum(s_x) > K:
            print("Error: More chargers at the station than admitted: {} chargers".format(sum(s_x)))

def installment_cost_check(my_plan, my_basic_cost):
    """
    check if instalment costs exceed budget
    """
    total_inst_cost = sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost
    if total_inst_cost > BUDGET:
        print("Error: Maximal BUDGET for installation costs exceeded.")


def control_charg_decision(my_plan, my_node_list):
    for p in range(P):
        for my_node in my_node_list:
            station_sum = sum([1 for my_station in my_plan if my_node[1]["charging station"][p] == my_station[0]])
            if station_sum > 1:
                print("Error: More than one station is assigned to a node.")
def waiting_time_check(my_plan):
    """
    check that wiating time is bounded
    """
    for my_station in my_plan:
        s_dict = my_station[2]
        for p in range(P):
            if s_dict["W_s"][p] == my_inf:
                print("Error: Waiting time goes to infinity.")


def constraint_check(my_plan, my_node_list, basic_cost):
    """
    test if solution satisfies all constraints
    """
    installment_cost_check(my_plan, basic_cost)
    control_charg_decision(my_plan, my_node_list)
    station_capacity_check(my_plan)
    waiting_time_check(my_plan)





