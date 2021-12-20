"""
The Benders decomposition algorithm for MDP
"""

# import
import time
import logging
import numpy as np
import gurobipy as grb


class MDP_Benders:
    """
    Benders Decomposition of Markov Decision Processes.
    """
    def __init__(
        self, name, MP, states, actions, trans_mat, reward_mat, gamma
    ):
        """
        `name`
        `MP`
        `MDP`
        Note that in this CLASS, s and a are indices.
        """
        super().__init__()
        self.start_time = time.time()
        self.final_time = "nan"
        self.name = name
        self.MP = MP
        self.state_dict = {
            s: states[s] for s in range(len(states))
        }
        self.action_dict = {
            a: actions[a] for a in range(len(actions))
        }
        self.trans_mat = trans_mat
        self.reward_mat = reward_mat
        self.gamma = gamma
        self.var_theta = {
            s: self.MP.getVarByName("theta_{}".format(s))
            for s in self.state_dict.keys()
        }
        self.var_vu = {}
        # ------------------------------------------------------
        # constructing all subproblems first
        self.subproblems, self.y_val = {}, {}
        for s in self.state_dict.keys():
            self.subproblems[s] = self.__build_MDP_dual(s, theta_val={
                s: -1e10
                for s in self.state_dict.keys()
            })
        # ------------------------------------------------------
        self.log = False

    def __build_MDP_dual(self, s, theta_val):
        """
        dual of MDP, single state
        `s`: key of current state
        `theta_val`
        """
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        # model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        var_y = {}
        for a in self.action_dict.keys():
            var_y[a] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="y_{}".format(a)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            grb.quicksum([
                self.reward_mat[s, a],
                grb.quicksum([
                    self.gamma * self.trans_mat[a][
                        s, s_new
                    ] * theta_val[s_new]
                    for s_new in self.state_dict.keys()
                ])
            ]) * var_y[a]
            for a in self.action_dict.keys()
        ])
        model.setObjective(objective, grb.GRB.MAXIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        model.addLConstr(
            lhs=grb.quicksum([
                var_y[a]
                for a in self.action_dict.keys()
            ]),
            sense=grb.GRB.EQUAL,
            rhs=1
        )
        model.update()
        return model

    def __modify_MDP_dual(self, s, theta_val):
        """
        dual of MDP, single state
        `s`: key of current state
        `theta_val`
        """
        # find variables and modify value
        for a in self.action_dict.keys():
            self.subproblems[s].getVarByName("y_{}".format(a)).setAttr(
                # self.var_y[s, a].setAttr(
                "Obj", np.sum([
                    self.reward_mat[s, a],
                    np.sum([
                        self.gamma * self.trans_mat[a][
                            s, s_new
                        ] * theta_val[s_new]
                        for s_new in self.state_dict.keys()
                    ])
                ])
            )
        self.subproblems[s].update()
        return

    def MDP_decomposition(self, sol_dir='None', write_log=False):
        """
        MDP Benders decomposition
        `sol_dir`: str, directory to output solution, do not include file name.
        `write_log`: bool, write solving data to log, defualt False.
        """
        self.log = write_log
        if self.log:
            logging.basicConfig(
                filename='{}.log'.format(self.name), filemode='w+',
                format='%(levelname)s - %(message)s', level=logging.INFO
            )
        # solving time of all models
        solve_time = 0
        # total time of the algorithm
        run_time = time.time()
        # start iteration
        iteration = 0
        while True:
            # solving first stage
            temp_time = time.time()
            self.MP.optimize()
            solve_time = solve_time + (time.time() - temp_time)
            # log
            if self.log:
                logging.info("=========================")
                logging.info("Iteration: {}".format(iteration))
            # check first stage feasibility
            if self.MP.status == grb.GRB.INFINITY:
                raise ValueError("First stage infeasible!")
                return
            elif self.MP.status == grb.GRB.OPTIMAL:
                # register first stage variables
                # theta
                theta_val = {
                    s: self.MP.getVarByName(
                        "theta_{}".format(s)
                    ).X
                    for s in self.state_dict.keys()
                }
                # log
                if self.log:
                    logging.info("MP Objective: {}".format(self.MP.ObjVal))
                    logging.info("theta: {}".format(theta_val))
            else:
                raise ValueError(
                    "First stage optimality code {}.".format(self.MP.status)
                )
            # going through each scenario
            value = {}
            optimal = True
            epsilon = 1e-5
            for s in self.state_dict.keys():
                # modify second stage dual
                self.__modify_MDP_dual(s, theta_val)
                # solve
                temp_time = time.time()
                self.subproblems[s].optimize()
                solve_time = solve_time + (time.time() - temp_time)
                # check status
                if self.subproblems[s].status == grb.GRB.UNBOUNDED:
                    # feasibility cut?
                    raise ValueError("Second stage {} infeasible!".format(s))
                elif self.subproblems[s].status == grb.GRB.INFEASIBLE:
                    raise ValueError("Second stage {} unbounded!".format(s))
                elif self.subproblems[s].status == grb.GRB.OPTIMAL:
                    # register solution
                    for a in self.action_dict.keys():
                        self.y_val[s, a] = self.subproblems[s].getVarByName(
                            "y_{}".format(a)
                        ).X
                    # log
                    if self.log:
                        logging.info("-------------------------")
                        logging.info(
                            "    State: {}".format(self.state_dict[s])
                        )
                        logging.info("    Objective: {}".format(
                            self.subproblems[s].ObjVal
                        ))
                    # e
                    e = np.dot([
                        self.y_val[s, a]
                        for a in self.action_dict.keys()
                    ], [
                        self.reward_mat[s, a]
                        for a in self.action_dict.keys()
                    ]
                    )
                    # coeff
                    coeff = {}
                    for s_new in self.state_dict.keys():
                        coeff[s_new] = (-1) * self.gamma * np.sum([
                            self.y_val[s, a] * self.trans_mat[a][s, s_new]
                            for a in self.action_dict.keys()
                        ])
                    # E
                    E = np.sum([
                        coeff[s_new] * theta_val[s_new]
                        for s_new in self.state_dict.keys()
                    ])
                    value[s] = e - E
                    # condition
                    if self.log:
                        logging.info("Optimality condition:")
                        logging.info(
                            "    State {}, theta = {}, value = {}".format(
                                self.state_dict[s],
                                self.var_theta[s].X, value[s]
                            )
                        )
                    if any([
                        theta_val[s] >= value[s],
                        np.abs(theta_val[s] - value[s]) <= epsilon,
                    ]) and iteration != 0:
                        # optimal, continue
                        continue
                    else:
                        optimal = False
                        # add optimality cut
                        self.MP.addLConstr(
                            lhs=grb.quicksum([self.var_theta[s]]),
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=e - grb.quicksum([
                                coeff[s_new] * self.var_theta[s_new]
                                for s_new in self.state_dict.keys()
                            ])
                        )
                    continue
            # after checking all states
            if optimal:
                self.final_time = time.time() - self.start_time
                # self.MP.write("model/{}-MPMDP.lp".format(self.name))
                if self.log:
                    logging.info("==============================")
                    logging.info("Final Objective: {}".format(self.MP.ObjVal))
                # stop loop
                break
            else:
                self.MP.update()
                # self.MP.write("model/{}-MPMDP.lp".format(self.name))
                iteration += 1
                continue
        # ------------------------- Output --------------------------
        run_time = time.time() - run_time
        # extract policy
        policy = {}
        for s in self.state_dict.keys():
            policy[self.state_dict[s]] = self.action_dict[np.argmax([
                self.y_val[s, a]
                for a in self.action_dict.keys()
            ])]
        # gap
        gap = 0 if self.MP.IsMIP == 0 else self.MP.MIPGap
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open(
                "{}/{}-Benders.txt".format(sol_dir, self.name), mode="w+"
            )
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            # optimal value and gap
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(self.MP.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for s in self.state_dict.keys():
                file.write("{}: {}\n".format(self.state_dict[s], value[s]))
            # optimal policy
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.state_dict.values():
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return self.MP.ObjVal, gap, run_time, solve_time, policy

    def __modify_MDP_dual_monotone(self, s, theta_val, best_action_ind):
        """
        dual of MDP, single state
        `s`: key of current state
        `theta_val`
        """
        # find variables and modify value
        for a in self.action_dict.keys():
            if a < best_action_ind:
                self.subproblems[s].getVarByName("y_{}".format(a)).setAttr(
                    "UB", 0
                )
            else:
                self.subproblems[s].getVarByName("y_{}".format(a)).setAttr(
                    "UB", grb.GRB.INFINITY
                )
            self.subproblems[s].getVarByName("y_{}".format(a)).setAttr(
                "Obj", np.sum([
                    self.reward_mat[s, a],
                    np.sum([
                        self.gamma * self.trans_mat[a][
                            s, s_new
                        ] * theta_val[s_new]
                        for s_new in self.state_dict.keys()
                    ])
                ])
            )
        self.subproblems[s].update()
        return

    def MDP_decomposition_monotone(self, sol_dir='None', write_log=False):
        """
        MDP Benders decomposition with monotone policy
        `sol_dir`: str, directory to output solution, do not include file name.
        `write_log`: bool, write solving data to log, defualt False.
        """
        self.log = write_log
        if self.log:
            logging.basicConfig(
                filename='{}.log'.format(self.name), filemode='w+',
                format='%(levelname)s - %(message)s', level=logging.INFO
            )
        # solving time of all models
        solve_time = 0
        # total time of the algorithm
        run_time = time.time()
        # start iteration
        iteration = 0
        all_policy = {}
        while True:
            all_policy[iteration] = {}
            # solving first stage
            temp_time = time.time()
            self.MP.optimize()
            solve_time = solve_time + (time.time() - temp_time)
            # log
            if self.log:
                logging.info("=========================")
                logging.info("Iteration: {}".format(iteration))
            # check first stage feasibility
            if self.MP.status == grb.GRB.INFINITY:
                raise ValueError("First stage infeasible!")
            elif self.MP.status == grb.GRB.OPTIMAL:
                # register first stage variables
                # theta
                theta_val = {
                    s: self.MP.getVarByName(
                        "theta_{}".format(s)
                    ).X
                    for s in self.state_dict.keys()
                }
                # log
                if self.log:
                    logging.info("MP Objective: {}".format(self.MP.ObjVal))
                    logging.info("theta: {}".format(theta_val))
            else:
                raise ValueError(
                    "First stage optimality code {}.".format(self.MP.status)
                )
            # going through each scenario
            value = {}
            optimal = True
            epsilon = 1e-5
            best_action_ind = 0
            for s in self.state_dict.keys():
                # modify second stage dual
                self.__modify_MDP_dual_monotone(s, theta_val, best_action_ind)
                # solve
                temp_time = time.time()
                self.subproblems[s].optimize()
                solve_time = solve_time + (time.time() - temp_time)
                # check status
                if self.subproblems[s].status == grb.GRB.UNBOUNDED:
                    # feasibility cut?
                    raise ValueError("Second stage {} infeasible!".format(s))
                elif self.subproblems[s].status == grb.GRB.INFEASIBLE:
                    raise ValueError("Second stage {} unbounded!".format(s))
                elif self.subproblems[s].status == grb.GRB.OPTIMAL:
                    # register solution
                    for a in self.action_dict.keys():
                        self.y_val[s, a] = self.subproblems[s].getVarByName(
                            "y_{}".format(a)
                        ).X
                    all_policy[iteration][s] = self.y_val[s, 1]
                    best_action_ind = np.argmax([
                        self.y_val[s, a] for a in self.action_dict.keys()
                    ])
                    # log
                    if self.log:
                        logging.info("-------------------------")
                        logging.info(
                            "    State: {}".format(self.state_dict[s])
                        )
                        logging.info("    Objective: {}".format(
                            self.subproblems[s].ObjVal
                        ))
                    # e
                    e = np.dot([
                        self.y_val[s, a]
                        for a in self.action_dict.keys()
                    ], [
                        self.reward_mat[s, a]
                        for a in self.action_dict.keys()
                    ]
                    )
                    # coeff
                    coeff = {}
                    for s_new in self.state_dict.keys():
                        coeff[s_new] = (-1) * self.gamma * np.sum([
                            self.y_val[s, a] * self.trans_mat[a][s, s_new]
                            for a in self.action_dict.keys()
                        ])
                    # E
                    E = np.sum([
                        coeff[s_new] * theta_val[s_new]
                        for s_new in self.state_dict.keys()
                    ])
                    value[s] = e - E
                    # condition
                    if self.log:
                        logging.info("Optimality condition:")
                        logging.info(
                            "    State {}, theta = {}, value = {}".format(
                                self.state_dict[s],
                                self.var_theta[s].X, value[s]
                            )
                        )
                    if any([
                        theta_val[s] >= value[s],
                        np.abs(theta_val[s] - value[s]) <= epsilon
                    ]):
                        # optimal, continue
                        continue
                    else:
                        optimal = False
                        # add optimality cut
                        self.MP.addLConstr(
                            lhs=grb.quicksum([self.var_theta[s]]),
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=e - grb.quicksum([
                                coeff[s_new] * self.var_theta[s_new]
                                for s_new in self.state_dict.keys()
                            ])
                        )
                    continue
            # after checking all states
            if optimal:
                self.final_time = time.time() - self.start_time
                # self.MP.write("model/{}-MPMDP.lp".format(self.name))
                if self.log:
                    logging.info("==============================")
                    logging.info("Final Objective: {}".format(self.MP.ObjVal))
                # stop loop
                break
            else:
                self.MP.update()
                # self.MP.write("model/{}-MPMDP.lp".format(self.name))
                iteration += 1
                continue
        # ------------------------- Output --------------------------
        run_time = time.time() - run_time
        # extract policy
        policy = {}
        for s in self.state_dict.keys():
            policy[self.state_dict[s]] = self.action_dict[np.argmax([
                self.y_val[s, a]
                for a in self.action_dict.keys()
            ])]
        # gap
        gap = 0 if self.MP.IsMIP == 0 else self.MP.MIPGap
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open(
                "{}/{}-Benders_mono.txt".format(sol_dir, self.name), mode="w+"
            )
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write(
                "Total model solving time: {} seconds;\n".format(solve_time)
            )
            # optimal value and gap
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(self.MP.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for s in self.state_dict.keys():
                file.write("{}: {}\n".format(self.state_dict[s], value[s]))
            # optimal policy
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.state_dict.values():
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return self.MP.ObjVal, gap, run_time, solve_time, policy
