import time
import math
import operator
import ast


class CalculatorError(Exception):
    pass


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}


def _safe_eval(node, depth=0, max_depth=50):
    if depth > max_depth:
        raise CalculatorError(f"Expression nesting too deep (max {max_depth})")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError("Invalid constant type")

    if isinstance(node, ast.Num):
        return node.n

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise CalculatorError(f"Operator {op_type.__name__} not allowed")

        left_val = _safe_eval(node.left, depth + 1, max_depth)
        right_val = _safe_eval(node.right, depth + 1, max_depth)

        if op_type == ast.Div and right_val == 0:
            raise ZeroDivisionError("Division by zero")

        if op_type == ast.Pow:
            if abs(right_val) > 1000:
                raise CalculatorError("Exponent too large")
            if abs(left_val) > 10**10 and right_val > 1:
                raise CalculatorError("Base too large for exponentiation")

        return _ALLOWED_OPERATORS[op_type](left_val, right_val)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise CalculatorError(f"Unary operator {op_type.__name__} not allowed")
        return _ALLOWED_OPERATORS[op_type](
            _safe_eval(node.operand, depth + 1, max_depth)
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only simple function calls allowed")

        fn_name = node.func.id
        if fn_name not in _ALLOWED_FUNCTIONS:
            raise CalculatorError(f"Function '{fn_name}' not allowed")

        args = [_safe_eval(arg, depth + 1, max_depth) for arg in node.args]

        if fn_name == "abs":
            if len(args) != 1:
                raise CalculatorError("abs() takes exactly 1 argument")
        if fn_name == "round":
            if not (1 <= len(args) <= 2):
                raise CalculatorError("round() takes 1 or 2 arguments")

        if fn_name == "sqrt" and args[0] < 0:
            raise ValueError("Cannot take square root of negative number")
        if fn_name == "log":
            if args[0] <= 0:
                raise ValueError("Logarithm requires positive argument")
        if fn_name == "exp" and args[0] > 700:
            raise OverflowError("Exponential argument too large")

        return _ALLOWED_FUNCTIONS[fn_name](*args)

    raise CalculatorError(f"Unsupported expression type: {type(node).__name__}")


def calculator(expr: str, max_len: int = 500, max_depth: int = 50) -> dict:

    t0 = time.perf_counter()

    if not isinstance(expr, str):
        return {
            "expr": str(expr)[:50],
            "result": None,
            "result_str": None,
            "status": "error",
            "error": "Expression must be a string",
            "elapsed_ms": 0,
        }

    expr = expr.strip()

    if len(expr) > max_len:
        return {
            "expr": expr[:50] + "...",
            "result": None,
            "result_str": None,
            "status": "error",
            "error": f"Expression too long (max {max_len} characters)",
            "elapsed_ms": 0,
        }


    if not expr:
        return {
            "expr": expr,
            "result": None,
            "result_str": None,
            "status": "error",
            "error": "Empty expression",
            "elapsed_ms": 0,
        }


    expr_norm = expr.replace("^", "**")

    try:
        tree = ast.parse(expr_norm, mode="eval")

        result = _safe_eval(tree.body, depth=0, max_depth=max_depth)

        if isinstance(result, float):
            if math.isnan(result):
                raise CalculatorError("Result is NaN (Not a Number)")
            if math.isinf(result):
                raise CalculatorError("Result is infinite")

        status = "ok"
        error = None

    except ZeroDivisionError:
        result = None
        status = "error"
        error = "Division by zero"

    except OverflowError as e:
        result = None
        status = "error"
        error = f"Numerical overflow: {str(e)}"

    except ValueError as e:
        result = None
        status = "error"
        error = f"Math domain error: {str(e)}"

    except SyntaxError as e:
        result = None
        status = "error"
        error = f"Invalid syntax: {str(e)}"

    except CalculatorError as e:
        result = None
        status = "error"
        error = str(e)

    except Exception as e:
        result = None
        status = "error"
        error = f"Unexpected error: {str(e)}"

    t1 = time.perf_counter()

    result_str = None if result is None else str(result)

    return {
        "expr": expr,
        "result": result,
        "result_str": result_str,
        "status": status,
        "error": error,
        "elapsed_ms": round((t1 - t0) * 1000.0, 3),
    }


if __name__ == "__main__":
    test_cases = [
        "2 + 3",
        "10 / 2",
        "sqrt(16)",
        "sin(3.14159 / 2)",
        "2 ** 10",
        "2^10",          
        "3^(2+1)",       
        "log(exp(5))",
        "1 / 0",         
        "sqrt(-1)",      
        "2 ** 10000",    
        "((((1+1)+1)+1))",  
        "",            
    ]

    print("Calculator Tool Test Results:")
    print("=" * 60)
    for expr in test_cases:
        out = calculator(expr)
        print(f"Expression: {out['expr']}")
        print(f"Result: {out['result']}")
        print(f"Result(str): {out['result_str']}")
        print(f"Status: {out['status']}")
        if out["error"]:
            print(f"Error: {out['error']}")
        print(f"Time: {out['elapsed_ms']:.3f}ms")
        print("-" * 60)