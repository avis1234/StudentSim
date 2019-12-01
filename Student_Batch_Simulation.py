import numpy as np
import pandas as pd
import scipy
from scipy import stats
import copy
from time import strftime
import matplotlib.pyplot as plt
import pickle
import os

# Helper functions

def randint_norepeat(max_val, size):
    """
    return a vector with length of size, of random integers between 0 and max_val-1, *with no repeats*
    Used to select random students ids from id list
    """
    size = int(size)
    assert max_val - size >= 1

    if (size == max_val - 1):
        return np.arange(0, max_val, dtype=int)
    else:
        items_so_far = 0
        items_vec = -1 * np.ones(size, dtype=int)
        while (items_so_far < size):
            item = np.random.randint(0, max_val, dtype=int)
            if item not in items_vec:
                items_vec[items_so_far] = item
                items_so_far += 1

        return items_vec


class StudentObj:
    """" Student class to be used in the simulation"""

    def __init__(self, model_name, student_id, num_skills=10, init_scheme='normal', skills_mu=0.6, skills_sigma=0.15, alpha=0.1, beta=0.2, gamma=1, eta=0.5):
        """
        :param model_name: student model name ['IRT', 'simple']
        :param student_id: int
        :param num_skills: number of different skills exists in simulation
        :param init_scheme: ['random', 'normal'] how to init students competency level of each skill
        :param skills_mu: expectation for normal mode init
        :param skills_sigma: std for normal mode init
        IRT parameters:
        :param alpha: skill update factor after the student answered a question
        :param beta: stochastic term's weight in grade
        :param gamma: scale of the deterministic term in grade (higher = more extreme values for the same competency level difference)
        :param eta: threshold for success in a question (grade > threshold will increase student's competency level in a certain skill)
        """
        if model_name not in ['IRT', 'simple']:
            raise ValueError

        self.model_name = model_name
        self.student_id = student_id
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        if (init_scheme == 'uniform'):
            self.skills_arr = np.random.random_sample(num_skills)
        elif (init_scheme == 'normal'):
            # Normal distribution will not do here, since it spreads on (-inf, inf).
            # we need normal distribution trunced between [0,1].
            lower = 0
            upper = 1
            self.skills_arr = scipy.stats.truncnorm.rvs((lower - skills_mu) / skills_sigma,
                                                        (upper - skills_mu) / skills_sigma,
                                                        loc=skills_mu, scale=skills_sigma, size=num_skills)
        elif isinstance(init_scheme, tuple):
            a = init_scheme[0]
            b = init_scheme[1]
            self.skills_arr = np.random.uniform(a, b, size=num_skills)


    def answer_question(self, q_id, skill_id, q_comp_lvl, change_skill=True):
        if type(q_id) != type(int):
            q_id = int(q_id)
        if (type(skill_id) != type(int)):
            skill_id = int(skill_id)
        # --- simple model ------------------------------------------
        if (self.model_name == 'simple'):
            relevant_skill = self.skills_arr[skill_id]
            if (relevant_skill + 0.1) > 1:
                self.skills_arr[skill_id] = 1
                return relevant_skill
            else:
                self.skills_arr[skill_id] = relevant_skill + 0.1
                return relevant_skill
        # -----------------------------------------------------------

        # --- IRT model ---------------------------------------------
        elif (self.model_name == 'IRT'):
            stu_comp_lvl = self.skills_arr[skill_id]

            deterministic_term = 1. / (1. + np.exp(self.gamma * (q_comp_lvl - stu_comp_lvl) ))
            stochastic_term = np.random.uniform(0, 1)
            grade = (1-self.beta)*deterministic_term + self.beta*stochastic_term

            # q_comp_factor is a factor to scale the skill change according to the difficdulty level of the question
            # compared to student skill. in case of success, we want harder question to result in larger posative
            # skill change.
            # in case of failure, we want harder question to result in smaller negative skill change.
            #alpha1, alpha2 = 0.1, 0.08
            alpha1, alpha2 = 0.15, 0.08
            if change_skill:
                if (grade > self.eta):
                    #q_comp_factor = 30**q_comp_lvl
                    q_comp_factor = alpha1*q_comp_lvl
                    stu_comp_lvl = stu_comp_lvl*(1-q_comp_factor) + q_comp_factor
                else:
                    #q_comp_factor = 10**(- q_comp_lvl)
                    #q_comp_factor = 0.08
                    q_comp_factor = (1-q_comp_lvl)*alpha2
                    stu_comp_lvl = stu_comp_lvl*(1-q_comp_factor)


            # update student competency level
                #stu_comp_lvl = stu_comp_lvl + self.alpha*q_comp_factor*(grade - self.eta)
                #stu_comp_lvl = (1-self.alpha) * stu_comp_lvl + self.alpha*q_comp_factor
                # fix to be in [0,1]
                if (stu_comp_lvl > 1):
                    stu_comp_lvl = 1
                elif (stu_comp_lvl < 0):
                    stu_comp_lvl = 0

                self.skills_arr[skill_id] = stu_comp_lvl

            return grade
        # -----------------------------------------------------------



class studentSimulation:
    """ Student Simulation Ojbect"""

    def __init__(self, data_dir, model_name="IRT", num_students=100, num_questions=300, num_skills=10,
                 init_scheme='normal', sim_name='StudentSim', skills_mu=0.6, skills_sigma=0.15,
                 IRT_params={'alpha':0.1, 'beta':0.2, 'gamma':1, 'eta':0.5}, verbose=False):
        self.data_dir = data_dir
        self.model_name = model_name
        self.model_name = model_name
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_skills = num_skills
        self.init_scheme = init_scheme
        self.sim_name = sim_name

        # Generate students as a list of student objects
        self.student_list = []
        if (init_scheme == 'groups'):
            num_stu_per_group = int(np.floor(float(num_students)/3))
            for i in range(num_stu_per_group):
                self.student_list.append(StudentObj(model_name, i, num_skills, (0, 0.3), skills_mu, skills_sigma, **IRT_params))
            for i in range(num_stu_per_group, 2*num_stu_per_group):
                self.student_list.append(StudentObj(model_name, i, num_skills, (0.3, 0.6), skills_mu, skills_sigma, **IRT_params))
            for i in range((num_stu_per_group*2), num_students):
                self.student_list.append(StudentObj(model_name, i, num_skills, (0.6, 0.9), skills_mu, skills_sigma, **IRT_params))
        else:
            for i in range(num_students):
                self.student_list.append(
                    StudentObj(model_name, i, num_skills, init_scheme, skills_mu, skills_sigma, **IRT_params))


        # for i in range(num_students):
        #     self.student_list.append(StudentObj(model_name, i, num_skills, init_scheme, skills_mu, skills_sigma, **IRT_params))

        # Generate questions as pandas data frame, and export to file
        self.questions_df = pd.DataFrame(index=np.arange(num_questions), columns=['q_id', 'skill', 'difficulty'])
        self.questions_df.q_id = np.arange(num_questions, dtype=int)
        self.questions_df.skill = np.random.randint(0, num_skills, size=num_questions, dtype=int)
        self.questions_df.difficulty = np.random.random_sample(num_questions)
        # TODO : check difficulty init. for now it's uniform [0,1]
        q_info_fname = data_dir + sim_name + '_questions_info.csv'
        self.questions_df.to_csv(q_info_fname, index=False)


        # Create students df and export to file
        cols = ['student_id'] + ['skill_{}'.format(i) for i in range(num_skills)]
        self.students_df = pd.DataFrame(index=np.arange(num_students), columns=cols)
        for i in range(num_students):
            self.students_df.iloc[i].student_id = i
            self.students_df.iloc[i, 1:(num_skills+1)] = self.student_list[i].skills_arr

        priors_fname = data_dir + sim_name + '_students_prior_skills.csv'
        self.students_df.to_csv(priors_fname, index=False)

        if (verbose):
            print('Started simulation {} at {}.'.format(sim_name, strftime("%H:%M:%S")))
            print('==========================================================')
            #print('num_students={}, num_questions={}, num_skills={}'.format(num_students, num_questions, num_skills))
            print('Number of students  = {}'.format(num_students))
            print('Number of questions = {}'.format(num_questions))
            print('Number of skills    = {}'.format(num_skills))
            print('Student model       = {}'.format(model_name))
            if (model_name == 'IRT'):
                print('IRT parameters      = {}'.format(IRT_params))
            print('==========================================================')
            print('Log dir                     = {}'.format(data_dir))
            print('Student initial skills file = {}'.format(priors_fname))
            print('Questions data file         = {}'.format(q_info_fname))
            print('==========================================================')
            print('')
            print('')



    def get_question_vec(self, num_questions, mode, mu=0.5, sigma=0.1, verbose=False):
        """A function that selects questions with normal / uniform distribution of difficulty level"""
        difficulty_vec = self.questions_df.difficulty.copy().as_matrix()
        if (num_questions > len(difficulty_vec)):
            raise ValueError

        if (mode == 'random'):
            #return np.random.randint(0, len(difficulty_vec), size=num_questions)
            return randint_norepeat(len(difficulty_vec), size=num_questions)

        # [a,b] mode
        elif isinstance(mode, tuple):
            assert len(mode) == 2
            a,b = mode
            assert 0 < a and a < 1
            assert 0 < b and b < 1
            assert b > a

            below_a = np.where(difficulty_vec < a)[0]
            between_a_b = np.where((difficulty_vec > a) & (difficulty_vec < b))[0]
            above_b = np.where(difficulty_vec > b)[0]

            # Generate num_q_vec - a list of 3 numbers of question to take from each category (below a, betweeen a and b, above b)
            num_q_vec = [int(np.floor(num_questions/3))]*2
            num_q_vec.append(num_questions - sum(num_q_vec))
            idx_qvec = np.cumsum(num_q_vec)

            assert len(below_a) > num_q_vec[0]
            assert len(between_a_b) > num_q_vec[1]
            assert len(above_b) > num_q_vec[2]

            # build question vec
            q_vec = -1 * np.ones(num_questions)
            idx_questions_below_a = randint_norepeat(len(below_a), size=num_q_vec[0])
            q_vec[0:idx_qvec[0]] = below_a[idx_questions_below_a]
            idx_questions_between_a_b = randint_norepeat(len(between_a_b), size=num_q_vec[1])
            q_vec[idx_qvec[0]:idx_qvec[1]] = between_a_b[idx_questions_between_a_b]
            idx_questions_above_b = randint_norepeat(len(above_b), size=num_q_vec[2])
            q_vec[idx_qvec[1]:] = above_b[idx_questions_above_b]

            return q_vec

        # Sample from the wanted distribution a vector of num_questions samples
        elif (mode == 'normal'):
            lower = 0
            upper = 1
            samp = scipy.stats.truncnorm.rvs((lower - mu) / sigma,
                                             (upper - mu) / sigma,
                                             loc=mu, scale=sigma, size=num_questions)

        elif (mode == 'uniform'):
            samp = np.random.uniform(0, 1, size=num_questions)

        else:
            raise ValueError


        q_vec = np.zeros(num_questions)
        for i in range(num_questions):
            # Select the question with difficaulty value closest to the sample drawn
            q_idx = np.argmin(np.abs(samp[i] - difficulty_vec))
            q_vec[i] = q_idx
            if verbose:
                print(
                    '{}: sample={:.2f}, selected={} with diff={:.2f}'.format(i, samp[i], q_idx, difficulty_vec[q_idx]))
            # Set big value to the difficulty of the selected question, so it wont be selected again
            difficulty_vec[q_idx] = -10

        return q_vec


    def runBatchSimulation(self, num_students='all', num_questions=60, mode='random', mu=0.5, sigma=0.1, verbose=False, plot_filename=True, time_idx=0, change_skill=True):
        if mode not in ['random', 'normal', 'uniform']:
            raise ValueError
        if num_questions > self.num_questions:
            raise ValueError

        print('[BatchSim]: Started batch sim at time {}.'.format(strftime("%H:%M")))
        print('[BatchSim]: num_students={}, num_questions={}, mode={}.'.format(num_students, num_questions, mode))
        #q_vec = np.zeros(num_questions)
        if (num_students == 'all'):
            students_vec = np.arange(0, self.num_students, dtype=int)
        elif num_students < self.num_students:
            students_vec = randint_norepeat(self.num_students, num_students)
        else:
            print('** Error** : runBatchSimulation(): ivalid value of num_students.')
            raise ValueError

        batchsim_results_df = pd.DataFrame(columns=['student_id', 'question_id', 'q_skill', 'q_comp_lvl', 'stu_comp_lvl', 'grade', 'time'])

        for stu_idx in students_vec:
            student_id = students_vec[stu_idx]
            q_vec = self.get_question_vec(num_questions, mode, mu, sigma)
            t = time_idx
            for q_idx in range(num_questions):
                q_id = q_vec[q_idx]
                q_skill = int(self.questions_df.loc[q_id, 'skill'].copy())
                q_comp_lvl = self.questions_df.loc[q_id, 'difficulty'].copy()
                student_comp_lvl_before = self.student_list[student_id].skills_arr[q_skill]
                student_skill_arr = self.student_list[student_id].skills_arr.copy()
                student_skill_ser = pd.Series(index=['skill_{}'.format(i) for i in range(len(student_skill_arr))], data=student_skill_arr)
                grade = self.student_list[student_id].answer_question(q_id, q_skill, q_comp_lvl, change_skill)
                # answer_question(self, q_id, skill_id, q_comp_lvl)
                student_comp_lvl_after = self.student_list[student_id].skills_arr[q_skill]

                tmpdict = {'student_id': student_id, 'question_id': q_id, 'q_skill': q_skill, 'q_comp_lvl': q_comp_lvl, 'stu_comp_lvl': student_comp_lvl_before, 'grade': grade , 'time': t}
                tmpser = pd.Series(tmpdict)
                tmpser = tmpser.append(student_skill_ser)
                batchsim_results_df = batchsim_results_df.append(tmpser, ignore_index=True)
                #self.batchsim_results_df.append([student_id, q_id, q_skill, grade ])

                if (verbose):
                    print('[BatchSim]: student_idx={}, student_id={}, q_idx={}, q_id={}, q_skill={}, q_comp_lvl={:.2f}, student_comp_lvl_before={:.2f}, grade={:.2f}, student_comp_lvl_after={:.2f}.'.format(
                            stu_idx, student_id, q_idx, q_id, q_skill, q_comp_lvl, student_comp_lvl_before, grade,
                            student_comp_lvl_after))

                t=t+1


            # update student's skills vector in simulation df
            self.students_df.iloc[student_id, 1:] = self.student_list[student_id].skills_arr

        # export to dataframe to csv
        batchsim_fname = self.data_dir + self.sim_name + '_student_batch_run_results' + strftime("_%d-%m_%H-%M") + '.csv'
        batchsim_results_df.to_csv(batchsim_fname)

        print('[BatchSim]: Simulation ended at time {}.'.format(strftime("%H:%M")))
        if (plot_filename):
            print('[BatchSim]: results filename = {}'.format(batchsim_fname))
        print('---')

        return batchsim_results_df


    def runPreTest(self, num_questions=15, mode='random', mu=0.5, sigma=0.1, verbose=False, plot_filename=True, time_idx=10):
        if not ( (mode in ['random', 'normal', 'uniform']) or (isinstance(mode, tuple)) ) :
            raise ValueError
        if num_questions > self.num_questions:
            raise ValueError

        q_vec = self.get_question_vec(num_questions, mode, mu, sigma)

        print('[PreTest]: Started PreTest sim at time {}.'.format(strftime("%H:%M")))
        print('[PreTest]: num_questions={}, mode={}.'.format(num_questions, mode))
        print('[PreTest]: q_vec = {}'.format(q_vec))

        pretest_results_df = pd.DataFrame(columns=['student_id', 'question_id', 'q_skill', 'q_comp_lvl', 'stu_comp_lvl', 'grade', 'time'])

        for student_id in range(self.num_students):
            t = time_idx
            for q_idx in range(num_questions):
                q_id = int(q_vec[q_idx])
                q_skill = int(self.questions_df.loc[q_idx, 'skill'].copy())
                q_comp_lvl = self.questions_df.loc[q_id, 'difficulty'].copy()
                student_comp_lvl_before = self.student_list[student_id].skills_arr[q_skill]
                student_skill_arr = self.student_list[student_id].skills_arr.copy()
                student_skill_ser = pd.Series(index=['skill_{}'.format(i) for i in range(len(student_skill_arr))],
                                              data=student_skill_arr)
                grade = self.student_list[student_id].answer_question(q_id, q_skill, q_comp_lvl)
                # answer_question(self, q_id, skill_id, q_comp_lvl)
                student_comp_lvl_after = self.student_list[student_id].skills_arr[q_skill]

                tmpdict = {'student_id': student_id, 'question_id': q_id, 'q_skill': q_skill, 'q_comp_lvl': q_comp_lvl,
                           'stu_comp_lvl': student_comp_lvl_before, 'grade': grade, 'time': t,
                           'student_skills_arr': student_skill_arr}
                #tmpdict = {'student_id': student_id, 'question_id': q_id, 'q_skill': q_skill, 'grade': grade }
                tmpser = pd.Series(tmpdict)
                tmpser = tmpser.append(student_skill_ser)
                pretest_results_df = pretest_results_df.append(tmpser, ignore_index=True)
                #pretest_results_df.append([student_id, q_id, q_skill, grade ])

                if (verbose):
                    print('[PreTest]: student_id={}, q_idx={}, q_id={}, q_skill={}, q_comp_lvl={:.2f}, student_comp_lvl_before={:.2f}, grade={:.2f}, student_comp_lvl_after={:.2f}.'.format(
                          student_id, q_idx, q_id, q_skill, q_comp_lvl, student_comp_lvl_before, grade, student_comp_lvl_after))
                t += 1

            # update student's skills vector in simulation df
            self.students_df.iloc[student_id, 1:] = self.student_list[student_id].skills_arr

        # export dataframe to csv
        pre_fname = self.data_dir + self.sim_name + '_pre_test_results' + strftime("_%d-%m_%H-%M") + '.csv'
        pretest_results_df.to_csv(pre_fname)
        print('[PreTest]: Simulation ended at time {}.'.format(strftime("%H:%M")))
        if (plot_filename):
            print('[PreTest]: results filename = {}'.format(pre_fname))
        print('---')

        # return questions vector - to be saved for post test
        return q_vec, pretest_results_df


# ------------------------------------------------------------------------------------------------------------------


def create_batch_simulation(sim_tag, num_students, num_q_for_batch, init_scheme):
    basedir = r'./Simulator/'
    simdir = basedir + sim_tag + '/'
    # Create simdir
    if not os.path.exists(simdir):
        os.makedirs(simdir)

    data_dir = simdir + 'data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    x = studentSimulation(data_dir=data_dir,
                          model_name='IRT',
                          num_students=num_students,
                          num_questions=1000,
                          num_skills=10,
                          verbose=True,
                          IRT_params = {'alpha': 0.1, 'beta': 0.2, 'gamma': 5, 'eta': 0.5},
                          sim_name='base_sim',
                          init_scheme=init_scheme
                          )
    print('')
    time_idx = 0

    # create batch file for algorithms training. Student skill is not advanced.
    batch_res1 = x.runBatchSimulation(num_students='all', num_questions=num_q_for_batch, mode='random', verbose=True,
                                      time_idx=time_idx, change_skill=False)
    time_idx += num_q_for_batch

    # run pretest on simulated data. Student skill is advanced.
    q_vec, pre_res = x.runPreTest(10, mode=(0.3,0.7), verbose=True, plot_filename=True, time_idx=time_idx)
    time_idx += 10

    print('======================================================')
    print('')
    print('Batch Simulation Done. simdir = {}'.format(simdir))

    #return time_idx
    # --------------------------------------------------------------------------------------------------


# -------------------- M A I N ---------------------------------------------

if __name__ == '__main__':

    # A max of 1000 questions and 10 skills are used in the simulation. Each question belongs to one skill.
    # num_students: how many users in simulation.
    # num_q_for_batch: how many questions out of the 1000 questions will be used for this batch.

    sim_tag = 'sim_groups_10'

    num_students = 10
    num_q_for_batch = 50
    init_scheme = 'groups'  # 'normal'

    create_batch_simulation(sim_tag, num_students, num_q_for_batch, init_scheme)

