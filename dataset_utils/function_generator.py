from random import choice, randint, choices, random
from sympy import Add, Mul
from sympy import Symbol, Rational
from sympy import cos, sin, tan, log, sinh, cosh, tanh, sqrt, asin, asinh, acos, acosh, atan, atanh, Pow, exp
from sympy import diff
from sympy.core.numbers import ComplexInfinity, Infinity, NegativeInfinity, Number, NaN, ImaginaryUnit

import sys
from time import time, strftime
from os.path import isfile

from multiprocessing import Queue, Process
from queue import Empty

def curr_time():
    return strftime("%m-%d_%H-%M-%S")

def divide(a, b):
    if b == 0:
        return 0
    ta, tb = type(a), type(b)
    if (ta != int and ta != float) or (tb != int and tb != float):
        return a/b
    return Rational(a) / Rational(b)

class TwoFunc():
    """class of function that takes two positional arguments"""
    def __init__(self, thefunc):
        self.thefunc = thefunc
    def __call__(self, x, y):
        return self.thefunc(x,y)

class NFunc():
    """class of function that takes an arbitrary number of positional arguments"""
    def __init__(self, thefunc):
        self.thefunc = thefunc
    def __call__(self, *args):
        return self.thefunc(*args)

def notcool(expr):
    return any(isinstance(expr, x) for x in (ComplexInfinity, Infinity, NegativeInfinity, ImaginaryUnit, NaN))

def unsupported_op(ex, op, disallowed_sequential):
    if isinstance(ex, Number) and ex <= 0 and (op == log or op == sqrt):
        return True
    if isinstance(ex, Number) and op in set([cos, sin, tan, sinh, cosh, tanh, asin, atan, acos, asinh, acosh, atanh]):
        return True
    if op==log and type(ex) == Pow:
        return True
    if disallowed_sequential != None:
        disallowed_for_op = disallowed_sequential.get(op, None)
        if disallowed_for_op == type(ex):
            return True
    return False

def unsupported_2op(arg1, arg2, op):
    if type(op) != TwoFunc:
        return False
    if op.thefunc == divide and arg2 == 0:
        return True
    if op.thefunc == Pow and (arg2 == 0 or arg2 == 1 or (isinstance(arg1, Number) and arg1 <= 0)):
        return True
    if arg2 == 0:
        if op.thefunc == Add or op.thefunc == Mul:
            return True
    return False

def random_expr(ops, ops_probs, atoms_groups, atoms_probs, disallowed_sequential, nsteps=5, recursed=False):
    def get_atom():
        if atoms_probs == None:
            return choice(choice(atoms_groups))
        else:
            return choice(choices(population=atoms_groups, weights=atoms_probs, k=1)[0])
    def get_op():
        if ops_probs == None:
            return choice(choice(ops))
        else:
            return choice(choices(population=ops, weights=ops_probs, k=1)[0])
    ex = 0
    while ex == 0:
        ex = get_atom()
    op = get_op()
    if type(op) == TwoFunc or type(op) == NFunc:
        at2 = get_atom()
        exx = op(ex, at2)
        if not notcool(exx):
            ex = exx
    i_step, step_attemps, expr_copy = 0, 0, None
    while i_step < nsteps and step_attemps < 5*nsteps:
        expr_copy = ex
        i_step += 1
        step_attemps += 1
        op = get_op()
        if type(op) == TwoFunc:
            if not recursed and random() < 0.5:
                at2 = random_expr(ops, ops_probs, atoms_groups, atoms_probs, disallowed_sequential, nsteps=nsteps-i_step, recursed=True)
            else:
                at2 = get_atom()
            if unsupported_2op(ex, at2, op):
                i_step -= 1
                continue
            ex = op(ex, at2)
        elif type(op) == NFunc:

            if op.thefunc == Add and not recursed and random() < 0.5:
                ats = tuple(random_expr(ops, ops_probs, atoms_groups, atoms_probs, disallowed_sequential, nsteps=nsteps-i_step, recursed=True) for _ in range(1, 4))
            elif op.thefunc == Mul and not recursed and random() < 0.5:
                ats = tuple(random_expr(ops, ops_probs, atoms_groups, atoms_probs, disallowed_sequential, nsteps=nsteps-i_step, recursed=True) for _ in range(1, 2))
            else:
                ats = [get_atom() for _ in range(1, 4)]
            ex = op(ex, *ats)
        else:
            if unsupported_op(ex, op, disallowed_sequential):
                i_step -= 1
                continue
            ex = op(ex)
        if notcool(ex):
            ex = expr_copy
            i_step -= 1
            continue
    return ex

def rem_constant_terms_if_sum(f, var):
    """turns something like 2x - 6 + sin(pi/2) into 2x"""
    if type(f) == Add:
        return Add(*(arg for arg in f.args if diff(arg, var) != 0))
    return f

def gen_random_expr_and_deriv_fromconfig(ops, ops_probs, atoms, atoms_probs, numops, x, disallowed_sequential):
    f = rem_constant_terms_if_sum(random_expr(ops, ops_probs, atoms, atoms_probs, disallowed_sequential, numops), x)
    if type(f) == int:
        return None, None
    if isinstance(f, NaN) or f.has(ImaginaryUnit) or f.has(ComplexInfinity) or f.has(NegativeInfinity):
        return None, None
    df = diff(f, x)
    if df == 0:
        return None, None
    if not isinstance(df, int):
        if isinstance(df, NaN) or df.has(ImaginaryUnit) or df.has(ComplexInfinity) or df.has(NegativeInfinity):
            return None, None
    return f, df

def gen_random_expr_and_deriv(numops):
    operations = []
    operations.append([TwoFunc(f) for f in [Add, Mul]])
    operations.append([NFunc(f) for f in [Add, Mul]])
    operations.append([lambda x: -x])
    operations.append([TwoFunc(f) for f in [divide]])
    operations.append([TwoFunc(f) for f in [Pow]])
    # the non-wrapped ones take just one argument
    operations.append([sqrt])
    operations.append([log, exp])
    operations.append([cos, sin, tan])
    operations.append([sinh, cosh, tanh, asin, atan, acos])
    operations.append([asinh, acosh, atanh])

    ops_probs = [ 20, 8, 10, # add-mul and neg prob
                  5, 3, # divide, pow
                  6, 4, # sqrt, log-exp
                  10, 5, # trig, hyp and inverse trig
                  2 # inverse hyp
    ]
    
    nums = [1,2,3,4,5]
    x = Symbol("x")
    atoms = [[*nums, *[-n for n in nums]], [x]]

    return gen_random_expr_and_deriv_fromconfig(operations, ops_probs, atoms, None, numops, x)

def output_to_file(file_path, num_examples=500_000, max_num_calls=1_000_000):
    t0 = time()
    all_examples = set()
    if isfile(file_path):
        with open(file_path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
            lines = [l for l  in lines if l != ""]
        print(f"loading {len(lines)} existing examples")
        all_examples.update(lines)
    current_list = []

    calls, new_gen = 0, 0
    while len(all_examples) < num_examples and calls < max_num_calls:
        calls += 1
        try:
            f, df = gen_random_expr_and_deriv(randint(2, 8))
        except Exception as e:
            print("caught exception while generating example:", e)
            continue
        if f == None or df == None:
            continue
        example = f"{df} ===> {f} + C"
        if example in all_examples:
            continue
        new_gen += 1
        all_examples.add(example)
        current_list.append(example)
        if len(current_list) >= 5000:
            with open(file_path, "a") as f:
                f.write("\n".join(current_list) + "\n")
            current_list = []
            print(f"{curr_time()}: saved {new_gen} new examples (total {len(all_examples)}) with {calls} calls, {time()-t0} s from the start ")
    if len(current_list) > 0:
            with open(file_path, "a") as f:
                f.write("\n".join(current_list) + "\n")
                current_list = []
    print(f"{curr_time()}: finished! {new_gen} new examples generated")

def get_disallowed_op_and_arg_combos():
    combos = {sin:asin, cos:acos, tan:atan,
              sinh:asinh, cosh:acosh, tanh:atanh,
              log:exp}
    combos.update({v:k for (k,v) in combos.items()})

    return combos

def gen_batch_expr_and_deriv(max_batch_size, min_ops, max_ops):
    operations = []
    operations.append([TwoFunc(f) for f in [Add, Mul]])
    operations.append([NFunc(f) for f in [Add, Mul]])
    operations.append([lambda x: -x])
    operations.append([TwoFunc(f) for f in [divide]])
    operations.append([TwoFunc(f) for f in [Pow]])
    operations.append([sqrt])
    operations.append([log, exp])
    operations.append([cos, sin, tan])
    operations.append([sinh, cosh, tanh, asin, atan, acos])
    operations.append([asinh, acosh, atanh])

    ops_probs = [ 20, 8, 10, # add-mul and neg prob
                  5, 3, # divide, pow
                  6, 4, # sqrt, log-exp
                  5, 2, # trig, hyp and inverse trig
                  1 # inverse hyp
    ]

    nums = [1,2,3,4,5]
    x = Symbol("x")
    atoms = [[*nums, *[-n for n in nums]], [x]]
    disallowed_sequential = get_disallowed_op_and_arg_combos()
    examples = [gen_random_expr_and_deriv_fromconfig(operations, ops_probs,
                                                     atoms, None,
                                                     randint(min_ops, max_ops), x, disallowed_sequential)
                                                     for _ in range(max_batch_size)]
    return [ex for ex in examples if ex != (None, None)]

def produce_examples(receiver_q: Queue, min_ops, max_ops, max_batch_size):
    while True:
        while True:
            try:
                examples = gen_batch_expr_and_deriv(max_batch_size, min_ops, max_ops)
            except Exception as e:
                print("retrying generation because of error:", e)
                continue
            break
        try:
            receiver_q.put(examples, block=True)
        except Exception as e:
            print(e, type(e))
            receiver_q.close()
            break

def output_to_file_async(file_path, num_examples=500_000, max_num_calls=1_000_000, new_file_path=None):
    max_batch_size, batch_timeout_sec = 50, 5 # max examples returned by producer process at once and the time to wait for them before considering the process stuck and restarting it
    restarts, max_restarts = 0, 1000

    t0 = time()
    all_examples = set()
    if isfile(file_path):
        with open(file_path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
            lines = [l for l  in lines if l != ""]
        print(f"loading {len(lines)} existing examples")
        all_examples.update(lines)
    current_list = []
    dest_file_path = new_file_path if new_file_path else file_path

    queue = Queue(2) # buffered queue so producer can send next batch when the previous one is being analyzed
    p = Process(target=produce_examples, args=(queue, 2, 8, max_batch_size))
    p.start()
 
    calls, new_gen = 0, 0
    while len(all_examples) < num_examples and calls < max_num_calls:
        calls += max_batch_size
        try:
            batch = queue.get(block=True, timeout=batch_timeout_sec)
        except Empty:
            print(f"{curr_time()}: process stuck, killing it ({restarts+1}. time)")
            p.terminate()
            p.join()

            restarts += 1
            if restarts > max_restarts:
                print("maximum process restarts. refusing to continue")
                break

            p = Process(target=produce_examples, args=(queue, 2, 8, max_batch_size))
            p.start()
            print(f"{curr_time()}: new process started")
            continue
        except KeyboardInterrupt:
            break
        try:
            for f, df in batch:
                if f == None or df == None:
                    continue
                example = f"{df} ===> {f} + C"
                if example in all_examples:
                    continue
                new_gen += 1
                all_examples.add(example)
                current_list.append(example)

            if len(current_list) >= 5000:
                with open(dest_file_path, "a") as f:
                    f.write("\n".join(current_list) + "\n")
                current_list = []
                print(f"{curr_time()}: saved {new_gen} new examples (total {len(all_examples)}) with {calls} calls, {time()-t0} s from the start ")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("exception:", e)
            continue

    if len(current_list) > 0:
            with open(dest_file_path, "a") as f:
                f.write("\n".join(current_list) + "\n")

    print(f"{curr_time()}: finished! {new_gen} new examples generated with {calls} calls (total {len(all_examples)}), {time()-t0} s from the start")
    p.terminate()
    print("process terminated")
    p.join()
    print("exiting")

if __name__ == "__main__":
    parameters = {}
    for word_ind, word in enumerate(sys.argv):
        if word.startswith("--") and len(word) > 2 and word_ind < len(sys.argv)-1:
            parameters[word[2:]] = sys.argv[word_ind+1]
    
    if "file_path" not in parameters:
        print("file_path option required")
        sys.exit(0)
    for x in ["num_examples", "max_num_calls"]:
        if x in parameters:
            try:
                parameters[x] = int(parameters[x])
            except ValueError:
                print(f"{parameters[x]} as value for {x} cannot be interpreted as integer")
                sys.exit(0)
            if parameters[x] <= 0:
                print(f"{x} must be positive")
                sys.exit(0)
    output_to_file_async(**parameters)