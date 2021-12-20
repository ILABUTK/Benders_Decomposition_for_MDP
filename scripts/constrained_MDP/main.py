"""
Main file to validate model assumptions.
"""

# import
import pickle
import numpy as np
import gurobipy as grb
import scipy.stats as st
from itertools import product
from MDP import Constrained_MDP
from MDP_Benders import Constrained_MDP_Benders


def define_MDP(name, params):
    """
    Define an MDP
    """
    # which one to define
    # ----------------- random -------------------------
    if "random" in name:
        n_states = int(params[0])
        n_actions = int(params[1])
        # states
        states = list(range(n_states))
        # actions
        actions = list(range(n_actions))
        # transition matrix
        trans_mat = {}
        for a in actions:
            trans_mat[a] = np.random.rand(len(states), len(states))
            # normalize
            trans_mat[a] = trans_mat[a] / trans_mat[a].sum(
                axis=1, keepdims=True
            )
        # reward matrix
        reward_mat = np.random.randint(
            low=0, high=100, size=(len(states), len(actions))
        )

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=40000, scale=60000, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.999
    # ----------------- queue --------------------------
    elif "queue" in name:
        """
        deFarias2003a:
        - state: number of jobs;
        - action: which rate of service to choose;
        - transition, reward: see paper
        """
        # parameters
        n_state = params[0]
        n_action = params[1]
        p = params[2]
        q = list(np.linspace(0, 1 - p, n_action))
        # states
        states = list(range(n_state))
        # actions
        actions = list(range(n_action))
        # transition matrix
        trans_mat = {}
        for a in actions:
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            for i in range(trans_mat[a].shape[0]):
                # 0
                if i == 0:
                    trans_mat[a][i, 0] = 1 - p
                    trans_mat[a][i, 1] = p
                    continue
                # last
                elif i == trans_mat[a].shape[0] - 1:
                    trans_mat[a][i, i - 1] = q[a]
                    trans_mat[a][i, i] = 1 - q[a]
                    continue
                # middle
                else:
                    # trans_mat[a][i, i - 1] = q[a]
                    # trans_mat[a][i, i] = p
                    # trans_mat[a][i, i + 1] = 1 - (q[a] + p)
                    trans_mat[a][i, i - 1] = (1 - p) * q[a]
                    trans_mat[a][i, i] = (1 - p) * (1 - q[a]) + p * q[a]
                    trans_mat[a][i, i + 1] = p * (1 - q[a])
                    continue
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for j in range(reward_mat.shape[1]):
                reward_mat[i, j] = -1 * states[i] - (
                    60 * q[j] * q[j] * q[j]
                )

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=40000, scale=60000, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.999
    # ----------------- bandit -------------------------
    elif "bandit" in name:
        """
        Bertsimas2016:
        - n_arm bandit machine, each has n_states states,
            total n_arm^n_states states;
        - transition drawn uniformly;
        - no action, no transition;
        """
        # parameters
        n_arm = int(params[0])
        n_states = int(params[1])
        # pr of each bandit at all state, increasing
        pr = {}
        for i in range(n_arm):
            pr[i] = np.sort(st.uniform.rvs(loc=0, scale=100, size=n_states))
            pr[i] = pr[i] / np.sum(pr[i])
        # states
        state_list = list(product(range(n_states), repeat=n_arm))
        states = list(range(len(state_list)))
        # actions
        actions = list(range(n_arm))
        # transition matrix
        trans_mat = {}
        for a in actions:
            # arm transit
            trans_arm = np.random.rand(n_states, n_states)
            # normalize
            trans_arm = trans_arm / trans_arm.sum(
                axis=1, keepdims=True
            )
            # matrix
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            for i in range(trans_mat[a].shape[0]):
                for j in range(trans_mat[a].shape[1]):
                    if all([
                        state_list[i][:a] == state_list[j][:a],
                        state_list[i][a + 1:] == state_list[j][a + 1:]
                    ]):
                        trans_mat[a][i, j] = trans_arm[
                            state_list[i][a], state_list[j][a]
                        ]
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for a in range(reward_mat.shape[1]):
                reward_mat[i, a] = (10 / n_states) * state_list[i][a]

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=40000, scale=60000, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.999
    # ----------------- inventory ----------------------
    elif "inventory" in name:
        """
        Lee2017, Puterman2014:
        - state: inventory of items;
        - action: how much to order;
        - transition, reward: based on stochastic demand;
        """
        # parameters
        n_inv = int(params[0]) + 1
        discount_factor = 0.999
        b = st.uniform.rvs(loc=10, scale=15, size=1)[0]
        K = st.uniform.rvs(loc=3, scale=5, size=1)[0]
        c = st.uniform.rvs(loc=5, scale=7, size=1)[0]
        h = st.uniform.rvs(loc=0.1, scale=0.2, size=1)[0]
        d = st.norm.rvs(loc=0.5 * n_inv, scale=0.1 * n_inv, size=1)[0]
        p = st.poisson.pmf(range(n_inv), mu=d)
        p = p / np.sum(p)
        q = []
        for i in range(len(p)):
            q.append(np.sum(p[i:]))
        # states
        inv_list = list(range(-1, n_inv))
        states = list(range(len(inv_list)))
        # actions
        actions = list(range(n_inv))
        # transition matrix
        trans_mat = {}
        for a in actions:
            # matrix
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            for i in range(trans_mat[a].shape[0]):
                # overflow
                if inv_list[i] + a >= n_inv or inv_list[i] == -1:
                    trans_mat[a][i, inv_list.index(-1)] = 1
                    continue
                # else, look at each possible future state
                for j in range(trans_mat[a].shape[1]):
                    if inv_list[j] == -1:
                        continue
                    # demand
                    demand = inv_list[i] + a - inv_list[j]
                    # demnad < 0
                    if demand < 0:
                        trans_mat[a][i, j] = 0
                    elif demand >= 0 and inv_list[j] > 0:
                        trans_mat[a][i, j] = p[demand]
                    elif demand >= 0 and inv_list[j] == 0:
                        trans_mat[a][i, j] = q[inv_list[i] + a]
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            if inv_list[i] == -1:
                continue
            for a in range(reward_mat.shape[1]):
                # overflow
                if inv_list[i] + actions[a] >= n_inv:
                    reward_mat[i, a] = -100000
                    continue
                # normal reward
                Oa = 0 if actions[a] == 0 else K + c * actions[a]
                reward_mat[i, a] = np.sum([
                    b * j * p[j]
                    for j in range(inv_list[i] + actions[a] - 1)
                ]) - Oa - h * (inv_list[i] + actions[a])

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states) - 1)
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        initial_distr = [0] + list(initial_distr)
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=40000, scale=60000, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.999
    # ----------------- replace ------------------------
    elif "replace" in name:
        """
        Modified Puterman2014:
        - machine with states, 0: best state, higher the worse;
        - actions: which degree of maintenance;
        - transition: larger action has a higher pr to restore
            the state to 0;
        - reward: larger action has higher cost.
        """
        # parameters
        n_states = int(params[0])
        n_actions = int(params[1])
        # states
        states = list(range(n_states))
        # actions, 0: no maintenance
        actions = list(range(n_actions + 1))
        # -------- transition matrix ----------
        # deterorate pr
        p_det = np.zeros(shape=(len(states), len(states)))
        for i in range(p_det.shape[0]):
            mu = st.uniform.rvs(loc=0, scale=1, size=1)[0]
            pr = st.norm.pdf(
                states[i:], loc=states[i] - mu, scale=(i + 1)
            )
            pr = pr / np.sum(pr)
            for j in range(p_det.shape[1]):
                if j >= i:
                    p_det[i, j] = pr[j - i]
        # restore pr
        p_res = {}
        pr = st.uniform.rvs(loc=1, scale=n_actions, size=n_actions)
        pr = pr / np.sum(pr)
        for a in range(1, n_actions + 1):
            p_res[a] = pr[a - 1]
        # print(p_res)
        # exit()
        # transition
        trans_mat = {}
        for a in actions:
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            # no maintenance
            if a == 0:
                for i in range(trans_mat[a].shape[0]):
                    for j in range(trans_mat[a].shape[1]):
                        trans_mat[a][i, j] = p_det[i, j]
            # maintenance
            else:
                for i in range(trans_mat[a].shape[0]):
                    for j in range(trans_mat[a].shape[1]):
                        # success
                        if j < i:
                            trans_mat[a][i, j] = p_res[a] * p_det[0, j]
                        # no idea
                        else:
                            trans_mat[a][i, j] = np.sum([
                                # success
                                p_res[a] * p_det[0, j],
                                # fail
                                (1 - p_res[a]) * p_det[i, j]
                            ])
        # ---------- reward matrix -----------
        C_rew = 0.5 * len(states)
        C_rep = 0.1 * len(states) * np.array(actions)
        C_opr = np.sort(st.uniform.rvs(
            loc=0, scale=0.75 * len(states), size=len(states)
        ))
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for j in range(reward_mat.shape[1]):
                reward_mat[i, j] = C_rew - C_rep[j] - C_opr[i]

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = [1 / n_states] * n_states
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=40000, scale=60000, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.999
    # ----------------- transmission -------------------
    elif "transmission" in name:
        """
        Modified Krishnamurthy2016:
        - state: how many packages left and current channel;
        - action: transmit a package with which efforts;
        - transition: larger action has a higher chance of
            successful transmission;
        - reward: holding cost of packages, larger action
            has a higher transmission cost.
        """
        # parameters
        n_channel = params[0]
        n_package = params[1]
        n_action = params[2]
        # states
        state_list = []
        for j in range(n_channel):
            for k in range(n_package + 1):
                state_list.append((j, k))
        states = list(range(len(state_list)))
        # actions
        actions = list(range(n_action + 1))
        # success rate
        mu = np.sort(
            st.uniform.rvs(loc=0, scale=1, size=n_action)
        )
        p = {}
        for a in range(1, n_action + 1):
            p[a] = st.norm.pdf(
                range(n_channel), loc=n_channel - 1, scale=1 / mu[a - 1]
            )
            p[a] = p[a] / np.sum(p[a])
        # transmission cost
        c_t = -1 * np.sort(
            np.abs(st.uniform.rvs(loc=5, scale=15, size=n_action))
        )
        # hold cost
        c_h = -1 * np.abs(st.uniform.rvs(loc=0, scale=5, size=1)[0])
        # transition matrix
        trans_mat = {}
        # channel transmission
        channel_trans = np.random.rand(n_channel, n_channel)
        # normalize
        channel_trans = channel_trans / channel_trans.sum(
            axis=1, keepdims=True
        )
        for a in actions:
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            # not transimitting
            if a == 0:
                for i in range(trans_mat[a].shape[0]):
                    # channel
                    for j in range(trans_mat[a].shape[1]):
                        if all([
                            # remaining item
                            state_list[i][1] == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ]
            # transmitting
            else:
                for i in range(trans_mat[a].shape[0]):
                    for j in range(trans_mat[a].shape[1]):
                        # no package left
                        if all([
                            # remaining item
                            state_list[i][1] == 0,
                            state_list[i][1] == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ]
                        # fail
                        elif all([
                            # remaining item
                            state_list[i][1] == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ] * (1 - p[a][state_list[i][0]])
                        # success
                        elif all([
                            # remaining item
                            state_list[i][1] - 1 == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ] * p[a][state_list[i][0]]
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for j in range(reward_mat.shape[1]):
                if actions[j] == 0:
                    reward_mat[i, j] = state_list[i][1] * c_h
                else:
                    reward_mat[i, j] = np.sum([
                        # package cost
                        state_list[i][1] * c_h,
                        # transmit cost
                        c_t[j - 1]
                    ])

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=40000, scale=60000, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.999
    # ================ model ===================
    model = Constrained_MDP(
        name=name,
        states=states,
        actions=actions,
        trans_func=trans_func,
        reward_func=reward_func,
        initial_distr=initial_distr,
        discount_factor=discount_factor,
        d=d, D=D
    )
    return model, {
        'states': states,
        'actions': actions,
        'trans_mat': trans_mat,
        'reward_mat': reward_mat,
        'alpha': initial_distr,
        'gamma': discount_factor,
        'd': d, 'D': D
    }


def define_MP(states, initial_distr, D):
    """
    Define the first stage methematical programming model.
    """
    state_dict = {
        i: states[i]
        for i in range(len(states))
    }
    # Gurobi model
    model = grb.Model()
    model.setParam("OutputFlag", False)
    model.setParam("IntFeasTol", 1e-9)
    # the model pay the highest attention to numeric coherency.
    model.setParam("NumericFocus", 3)
    model.setParam("DualReductions", 0)
    # ----------------------- Variables -------------------------
    # theta
    var_theta = {}
    for s in state_dict.keys():
        var_theta[s] = model.addVar(
            lb=-1e10, ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS, name="theta_{}".format(s)
        )
    model.update()
    # theta
    var_rho = {}
    for i in range(len(D)):
        var_rho[i] = model.addVar(
            lb=0, ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS, name="rho_{}".format(i)
        )
    model.update()
    # objective
    objective = grb.quicksum([
        grb.quicksum([
            initial_distr[s] * var_theta[s]
            for s in state_dict.keys()
        ]),
        grb.quicksum([
            var_rho[i] * D[i]
            for i in range(len(D))
        ])
    ])
    model.setObjective(objective, grb.GRB.MINIMIZE)
    # ---------------------- Constraints ------------------------
    model.update()
    return model


def decomposition():
    """
    Decomposition of MDP
    """
    instance = "transmission"
    params = (20, 20, 400)
    LP_obj, LP_run, LP_sol, LP_policy = {}, {}, {}, {}
    LPD_obj, LPD_run, LPD_sol, LPD_policy = {}, {}, {}, {}
    Benders_obj, Benders_run, Benders_sol, Benders_policy = {}, {}, {}, {}
    # name
    instance_name = instance
    for s in params:
        instance_name = instance_name + "-{}".format(s)
    # ------------------------------------------------------------
    for i in range(1):
        print(i)
        name = instance_name + "-{}".format(i)
        # ------------------------------------------------------------
        # MDP model
        MDP_model, elements = define_MDP(
            name, params
        )
        # MP model
        MP_model = define_MP(
            MDP_model.states, MDP_model.initial_distr, MDP_model.D
        )
        print(len(MDP_model.states))
        print(len(MDP_model.actions))
        return
        # ------------------------------------------------------------
        # LP benchamark
        print("LP starting...")
        LP_obj[i], _, LP_run[i], LP_sol[i], LP_policy[i] = MDP_model.LP_dual(
          sol_dir="results"
        )
        return
        print(LP_obj[i], LP_run[i], LP_sol[i])
        # ------------------------------------------------------------
        # LP dual benchamark
        LPD_obj[i], _, LPD_run[i], LPD_sol[i],\
            LPD_policy[i] = MDP_model.LP_dual(
                sol_dir="results"
            )
        print(LPD_obj[i], LPD_run[i], LPD_sol[i])
        # ------------------------------------------------------------
        # two-stage model
        model = Constrained_MDP_Benders(
            name=name, MP=MP_model, states=elements['states'],
            actions=elements['actions'], trans_mat=elements['trans_mat'],
            reward_mat=elements['reward_mat'], gamma=elements['gamma'],
            d=elements['d'], D=elements['D']
        )
        # run algorithm
        Benders_obj[i], _, Benders_run[i], Benders_sol[i],\
            Benders_policy[i] = model.MDP_decomposition(
                sol_dir="results", write_log=False
            )
        print(Benders_obj[i], Benders_run[i], Benders_sol[i])
    # ------------------------------------------------------------
    pickle.dump({
        'LP_obj': LP_obj,
        'LP_run': LP_run,
        'LP_sol': LP_sol,
        'LP_policy': LP_policy,
        'LPD_obj': LPD_obj,
        'LPD_run': LPD_run,
        'LPD_sol': LPD_sol,
        'LPD_policy': LPD_policy,
        'Benders_obj': Benders_obj,
        'Benders_run': Benders_run,
        'Benders_sol': Benders_sol,
        'Benders_policy': Benders_policy
    }, open('results/statistics/{}.pickle'.format(instance_name), 'wb'))
    return


def main():
    """main"""
    np.random.seed(1)
    # decomposition
    decomposition()
    return


if __name__ == "__main__":
    main()
