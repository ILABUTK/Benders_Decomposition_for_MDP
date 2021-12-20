"""
MDP problem definition as a class
with value iteration algorithm
"""

# import
import time
import numpy as np
import gurobipy as grb
from copy import deepcopy as dcopy


class MDP:
    """MDP problem class"""

    def __init__(
        self, name, states, actions, trans_func,
        reward_func, initial_distr, discount_factor
    ):
        """
        `name`: str, name of the MDP;
        `states`: list, states;
        `actions`: list, actions;
        `trans_func`: function, the transition function,
            input: (new_state, old_state, action), output: pr;
        `reward_func`: function, the reward function,
            input: (state, action), output: number;
        `initial_distr`: list, initial distribution of states;
        `discount_factor`: numeric, discount factor, < 1.
        """
        super().__init__()
        self.name = name
        self.states = states
        self.actions = actions
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.initial_distr = initial_distr
        self.discount_factor = discount_factor

    # value iteration
    def VI(self, epsilon, sol_dir='None'):
        """
        Value iteration, MDP problems. Using in-place value updates.
        """
        # time
        run_time = time.time()
        # initialization
        threshold = epsilon
        epoch = 0
        value = {
            state: 0
            for state in self.states
        }
        # iteration
        while True:
            old_value = dcopy(value)
            # calculate new values
            for state in self.states:
                value[state] = np.max([
                    self.reward_func(state, action) + np.sum([
                        self.discount_factor * self.trans_func(
                            n_state, state, action
                        ) * value[n_state]
                        for n_state in self.states
                    ])
                    for action in self.actions
                ])
            # value difference between two iterations
            difference = np.max([
                np.absolute(value[state] - old_value[state])
                for state in self.states
            ])
            # check optimality condition
            if difference < threshold:
                break
            else:
                # entering next period
                epoch = epoch + 1
                continue
        # total value
        total_value = np.dot(self.initial_distr, list(value.values()))
        # Finding the best policy
        policy = {}
        for state in self.states:
            policy[state] = self.actions[np.argmax([
                self.reward_func(state, action) + np.sum([
                    self.discount_factor * self.trans_func(
                        n_state, state, action
                    ) * value[n_state]
                    for n_state in self.states
                ])
                for action in self.actions
            ])]
        # time
        run_time = time.time() - run_time
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}_VI.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write("Optimality reached at epoch {};\n".format(epoch))
            file.write("Total running time: {} seconds;\n".format(run_time))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(total_value))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.close()
        return policy, total_value

    # linear programming, dual
    def LP_dual(self, sol_dir='None'):
        """
        Solving using linear programming, dual formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        var_x = {}
        for s in state_dict.keys():
            for a in action_dict.keys():
                var_x[s, a] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="x_{}_{}".format(s, a)
                )
        model.update()
        # objective
        objective = grb.quicksum([
            self.reward_func(self.states[s], self.actions[a]) * var_x[s, a]
            for s in state_dict.keys()
            for a in action_dict.keys()
        ])
        model.setObjective(objective, grb.GRB.MAXIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        for s in state_dict.keys():
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        var_x[s, a]
                        for a in action_dict.keys()
                    ]),
                    -1 * grb.quicksum([
                        self.discount_factor * self.trans_func(
                            self.states[s], self.states[s_old], self.actions[a]
                        ) * var_x[s_old, a]
                        for s_old in state_dict.keys()
                        for a in action_dict.keys()
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=self.initial_distr[s],
                name="constr_{}".format(s)
            )
            model.update()
        # ------------------------ Solving --------------------------
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        run_time = time.time() - run_time
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                var_x[s, a].X for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = model.getConstrByName(
                "constr_{}".format(s)
            ).Pi
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LPD.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return model.ObjVal, gap, run_time, solve_time, policy

    # linear programming, primal
    def LP(self, sol_dir='None'):
        """
        Solving using linear programming, primal formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        var_v = {}
        for s in state_dict.keys():
            var_v[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="v_{}".format(s)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            self.initial_distr[s] * var_v[s]
            for s in state_dict.keys()
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        for s in state_dict.keys():
            for a in action_dict.keys():
                model.addLConstr(
                    lhs=grb.quicksum([
                        var_v[s],
                        -1 * grb.quicksum([
                            self.discount_factor * self.trans_func(
                                self.states[s_new], self.states[s],
                                self.actions[a]
                            ) * var_v[s_new]
                            for s_new in state_dict.keys()
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.reward_func(self.states[s], self.actions[a]),
                    name="constr_{}_{}".format(s, a)
                )
        model.update()
        # ------------------------ Solving --------------------------
        # model.write("model/{}-LP.lp".format(self.name))
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        run_time = time.time() - run_time
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                model.getConstrByName(
                    "constr_{}_{}".format(s, a)
                ).Pi
                for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = var_v[s].X
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LP.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return model.ObjVal, gap, run_time, solve_time, policy


class Constrained_MDP:
    """MDP problem class with constraints"""

    def __init__(
        self, name, states, actions, trans_func,
        reward_func, initial_distr, discount_factor,
        d, D
    ):
        """
        `name`: str, name of the MDP;
        `states`: list, states;
        `actions`: list, actions;
        `trans_func`: function, the transition function,
            input: (new_state, old_state, action), output: pr;
        `reward_func`: function, the reward function,
            input: (state, action), output: number;
        `initial_distr`: list, initial distribution of states;
        `discount_factor`: numeric, discount factor, < 1;
        `d`: dict, (i, |S|, |A|), coefficient of dual variable,
            i is the number of constraints;
        `D`: list, ith element is the rhs of the ith constraint.
        """
        super().__init__()
        self.name = name
        self.states = states
        self.actions = actions
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.initial_distr = initial_distr
        self.discount_factor = discount_factor
        self.d, self.D = d, D

    # linear programming, dual
    def LP_dual(self, sol_dir='None'):
        """
        Solving using linear programming, dual formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        var_x = {}
        for s in state_dict.keys():
            for a in action_dict.keys():
                var_x[s, a] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="x_{}_{}".format(s, a)
                )
        model.update()
        # objective
        objective = grb.quicksum([
            self.reward_func(self.states[s], self.actions[a]) * var_x[s, a]
            for s in state_dict.keys()
            for a in action_dict.keys()
        ])
        model.setObjective(objective, grb.GRB.MAXIMIZE)
        # ---------------------- Constraints ------------------------
        # the MDP constraint
        for s in state_dict.keys():
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        var_x[s, a]
                        for a in action_dict.keys()
                    ]),
                    -1 * grb.quicksum([
                        self.discount_factor * self.trans_func(
                            self.states[s], self.states[s_old], self.actions[a]
                        ) * var_x[s_old, a]
                        for s_old in state_dict.keys()
                        for a in action_dict.keys()
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=self.initial_distr[s],
                name="constr_{}".format(s)
            )
        model.update()
        # other constraints
        for i in range(len(self.D)):
            model.addLConstr(
                lhs=grb.quicksum([
                    self.d[i, s, a] * var_x[s, a]
                    for s in state_dict.keys()
                    for a in action_dict.keys()
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=self.D[i],
                name="dD_{}".format(s)
            )
        model.update()
        # ------------------------ Solving --------------------------
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        run_time = time.time() - run_time
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                var_x[s, a].X for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = model.getConstrByName(
                "constr_{}".format(s)
            ).Pi
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LPD.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return model.ObjVal, gap, run_time, solve_time, policy

    # linear programming, primal
    def LP(self, sol_dir='None'):
        """
        Solving using linear programming, primal formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # value, v
        var_v = {}
        for s in state_dict.keys():
            var_v[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="v_{}".format(s)
            )
        model.update()
        # rho, additional constraints
        var_rho = {}
        for i in range(len(self.D)):
            var_rho[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="rho_{}".format(i)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            grb.quicksum([
                self.initial_distr[s] * var_v[s]
                for s in state_dict.keys()
            ]),
            grb.quicksum([
                self.D[i] * var_rho[i]
                for i in range(len(self.D))
            ])
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        for s in state_dict.keys():
            for a in action_dict.keys():
                model.addLConstr(
                    lhs=grb.quicksum([
                        var_v[s],
                        -1 * grb.quicksum([
                            self.discount_factor * self.trans_func(
                                self.states[s_new], self.states[s],
                                self.actions[a]
                            ) * var_v[s_new]
                            for s_new in state_dict.keys()
                        ]),
                        grb.quicksum([
                            self.d[i, s, a] * var_rho[i]
                            for i in range(len(self.D))
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.reward_func(self.states[s], self.actions[a]),
                    name="constr_{}_{}".format(s, a)
                )
        model.update()
        # ------------------------ Solving --------------------------
        # model.write("model/{}-LP.lp".format(self.name))
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        run_time = time.time() - run_time
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                model.getConstrByName(
                    "constr_{}_{}".format(s, a)
                ).Pi
                for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = var_v[s].X
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LP.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            for i in range(len(self.D)):
                file.write("rho {}: {}\n".format(i, var_rho[i].X))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return model.ObjVal, gap, run_time, solve_time, policy
