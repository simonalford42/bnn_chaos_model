import re
import numpy as np
np.cube = lambda x: x**3
import pickle
import spock_reg_model

def strip_outer_parentheses(expr):
    expr = expr.strip()
    changed = True
    while changed:
        changed = False
        if expr.startswith("(") and expr.endswith(")"):
            count = 0
            for i, ch in enumerate(expr):
                if ch == '(':
                    count += 1
                elif ch == ')':
                    count -= 1
                if count == 0 and i < len(expr)-1:
                    # Found a closing parent before the end,
                    # so these aren't full wrapping parentheses.
                    break
            else:
                # If we completed the loop without breaking,
                # outer parentheses wrap the entire expr.
                expr = expr[1:-1].strip()
                changed = True
    return expr


def extract_top_level_operator(expr):
    # At this point, expr should have no fully wrapping outer parentheses.
    count = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            count += 1
        elif ch == ')':
            count -= 1
        elif count == 0 and ch in ['+', '-', '*', '/']:
            return i, ch
    return None, None


def split_expression(expr):
    # Strip fully wrapping parentheses first
    expr = strip_outer_parentheses(expr)

    idx, op = extract_top_level_operator(expr)
    if idx is None:
        return expr, None, None
    left = expr[:idx].strip()
    right = expr[idx+1:].strip()

    # If the left part has an extra '(' at the start without a matching closing ')'
    if left.startswith('(') and left.count('(') > left.count(')'):
        left = left[1:].strip()

    # If the right part has an extra ')' at the end without a matching opening '('
    if right.endswith(')') and right.count(')') > right.count('('):
        right = right[:-1].strip()

    return left, op, right


def parse_left_right(expr):
    left_expr, op, right_expr = split_expression(expr)
    return left_expr, right_expr


def transform_expression(subexpr, var_names):
    code = subexpr
    code = code.replace('^', '**')

    changed = True
    while changed:
        old_code = code
        # Convert functions to numpy equivalents using a negative lookbehind
        # to ensure we don't re-match already converted np.<func>
        # easiest way to deal with cube is just to create a np.cube function in the namespace
        code = re.sub(r'(?<!\.)\bcube\((.*?)\)', r'np.cube(\1)', code)
        code = re.sub(r'(?<!\.)\bcbrt\((.*?)\)', r'np.cbrt(\1)', code)
        code = re.sub(r'(?<!\.)\bsquare\((.*?)\)', r'np.square(\1)', code)
        code = re.sub(r'(?<!\.)\bcos\((.*?)\)', r'np.cos(\1)', code)
        code = re.sub(r'(?<!\.)\bsqrt\((.*?)\)', r'np.sqrt(\1)', code)
        code = re.sub(r'(?<!\.)\bexp\((.*?)\)', r'np.exp(\1)', code)
        code = re.sub(r'(?<!\.)\blog\((.*?)\)', r'np.log(\1)', code)
        changed = (code != old_code)

    # for v in var_names:
        # code = code.replace(v, f'x["{v}"]')

    # Build a pattern that matches any of the variable names as whole words
    # this way, we don't accidentally replace parts of variable names with other variable names
    var_pattern = r'\b(' + '|'.join(re.escape(v) for v in var_names) + r')\b'
    code = re.sub(var_pattern, lambda m: f'x["{m.group(1)}"]', code)

    return code


def make_eval_function(code):
    def f(x):
        return eval(code, {"np": np}, {"x": x})
    return f


def eval_pure_sr_expression(expr: str, var_names):
    '''
    expr comes from reg.equations_.iloc[-1].equation, e.g. "((square((cbrt(0.9319145) * cbrt(cube(cos_Omega2))) / square(cbrt(-1.6076273))) * square(-1.6076273)) * (square(square(-1.6076273)) + (-0.1881028 * e1)))"
    '''
    left_expr, right_expr = parse_left_right(expr)
    left_code = transform_expression(left_expr, var_names)
    right_code = transform_expression(right_expr, var_names)
    left_func = make_eval_function(left_code)
    right_func = make_eval_function(right_code)
    return left_func, right_func


def example():
    # Example usage:
    expr = "((square((cbrt(0.9319145) * cbrt(cube(cos_Omega2))) / square(cbrt(-1.6076273))) * square(-1.6076273)) * (square(square(-1.6076273)) + (-0.1881028 * e1)))"
    left_expr, right_expr = parse_left_right(expr)
    left_code = transform_expression(left_expr, ["cos_Omega2", "e1"])
    right_code = transform_expression(right_expr, ["cos_Omega2", "e1"])
    left_func = make_eval_function(left_code)
    right_func = make_eval_function(right_code)
    data = {"cos_Omega2": np.array([0.1,0.2]), "e1": np.array([1.0,2.0])}

    print("Left code:", left_code)
    print("Right code:", right_code)
    left_vals = left_func(data)
    right_vals = right_func(data)
    print("Left vals:", left_vals)
    print("Right vals:", right_vals)


def lambdify_pure_sr_expression(expr: str, var_names):
    '''
    expr is the result of reg.equations_.iloc[-1].equation or similar
    returns function which takes in (B, 100, 41) of X and returns (B, ) of predictions
    '''
    print('Original expression: ', expr)
    left_expr, right_expr = parse_left_right(expr)
    left_code = transform_expression(left_expr, var_names)
    right_code = transform_expression(right_expr, var_names)
    print("Left code:", left_code)
    print("Right code:", right_code)

    def f(x: np.ndarray):
        # x (B, 100, 41)
        x_every_10th = x[:, ::10].numpy()
        f1_inp = {v: x_every_10th[..., i] for i, v in enumerate(var_names)}
        f1_out = eval(left_code, {"np": np}, {"x": f1_inp})  # (B, T)
        # take std across T axis
        std = np.std(f1_out, axis=1) # (B, )
        f2_inp = {v: std for v in var_names}
        f2_out = eval(right_code, {"np": np}, {"x": f2_inp})  # (B, )
        return f2_out

    return f


def example2():
    reg = pickle.load(open('sr_results/32834.pkl', 'rb'))
    eq = reg.equations_.iloc[-1]
    var_names = reg.feature_names_in_
    expr = eq.equation
    left_expr, right_expr = parse_left_right(expr)
    left_code = transform_expression(left_expr, var_names)
    right_code = transform_expression(right_expr, var_names)

    model = spock_reg_model.load(24880)
    model.make_dataloaders()
    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    x_every_10th = x[:, ::10].numpy()
    f1_inp = {v: x_every_10th[..., i] for i, v in enumerate(var_names)}
    f1_out = eval(left_code, {"np": np}, {"x": f1_inp})  # (B, T)
    # take std across T axis
    std = np.std(f1_out, axis=1) # (B, )
    f2_inp = {v: std for v in var_names}
    f2_out = eval(right_code, {"np": np}, {"x": f2_inp})  # (B, )


def evaluate_pure_sr_model(version, plot_random=False):
    model = spock_reg_model.load(24880)
    model.make_dataloaders()
    from copy import deepcopy as copy
    if plot_random:
        assert model.ssX is not None
        tmp_ssX = copy(model.ssX)
        model.make_dataloaders(
            ssX=model.ssX,
            train=False,
            plot_random=True) #train=False means we show the whole dataset (assuming we don't train on it!)

        assert np.all(tmp_ssX.mean_ == model.ssX.mean_)

    val_dataloader = model._val_dataloader

    reg = pickle.load(open(f'sr_results/{version}.pkl', 'rb'))
    pure_results = {}

    for i in range(len(reg.equations_)):
        try:
            expr = reg.equations_.iloc[i].equation
            complexity = reg.equations_.iloc[i].complexity
            var_names = reg.feature_names_in_
            expr_fn = lambdify_pure_sr_expression(expr, var_names)

            mses = []
            for x, y in iter(val_dataloader):
                pred = expr_fn(x)
                mse = np.mean((pred[:, None] - y.numpy())**2)
                mses.append(mse)

            avg_mse = np.mean(mses)
            print(f'Complexity {complexity}, Avg mse:', avg_mse)
            pure_results[complexity] = avg_mse

        except Exception as e:
            print(f'Failed to evaluate expression {expr}: {e}')
            import traceback
            traceback.print_exc()
            continue

    pickle.dump(pure_results, open(f'pure_results.pkl', 'wb'))


if __name__ == '__main__':
    evaluate_pure_sr_model(81050, plot_random=False)