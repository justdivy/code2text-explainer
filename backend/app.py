import ast
import math

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20_000)


MODEL_NAME = "Salesforce/codet5-base-multi-sum"
MAX_INPUT_TOKENS = 512


def explain_arithmetic_return(code: str) -> str | None:
    """Describe simple Python return expressions from their actual operators."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    operations = {
        ast.Add: lambda left, right: f"adds {left} and {right}",
        ast.Sub: lambda left, right: f"subtracts {right} from {left}",
        ast.Mult: lambda left, right: f"multiplies {left} by {right}",
        ast.Div: lambda left, right: f"divides {left} by {right}",
        ast.FloorDiv: lambda left, right: f"divides {left} by {right} and rounds down",
        ast.Mod: lambda left, right: f"finds the remainder when {left} is divided by {right}",
        ast.Pow: lambda left, right: f"raises {left} to the power of {right}",
    }

    def find_return(statements):
        for statement in statements:
            if isinstance(statement, ast.Return):
                value = statement.value
                if isinstance(value, ast.BinOp) and type(value.op) in operations:
                    left = ast.unparse(value.left)
                    right = ast.unparse(value.right)
                    return f"The function {operations[type(value.op)](left, right)} and returns the result."
            elif isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                result = find_return(statement.body) or find_return(statement.orelse)
                if result:
                    return result
            elif isinstance(statement, (ast.With, ast.AsyncWith, ast.Try)):
                result = find_return(statement.body)
                if result:
                    return result
                for handler in getattr(statement, "handlers", []):
                    result = find_return(handler.body)
                    if result:
                        return result
                result = find_return(getattr(statement, "orelse", []))
                if result:
                    return result
                result = find_return(getattr(statement, "finalbody", []))
                if result:
                    return result
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result = find_return(statement.body)
                if result:
                    return result
        return None

    return find_return(tree.body)


def explain_permutations(code: str) -> str | None:
    """Explain common itertools.permutations examples using their actual arguments."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    function_names = set()
    module_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "itertools":
            function_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "permutations"
            )
        elif isinstance(node, ast.Import):
            module_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "itertools"
            )

    data_values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    data_values[target.id] = value

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    permutation_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        is_permutations = (
            isinstance(function, ast.Name) and function.id in function_names
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "permutations"
            and isinstance(function.value, ast.Name)
            and function.value.id in module_names
        )
        if is_permutations and node.args and isinstance(node.args[0], ast.Name):
            source_name = node.args[0].id
            if source_name in data_values and hasattr(data_values[source_name], "__len__"):
                length = None
                if len(node.args) > 1:
                    try:
                        length = ast.literal_eval(node.args[1])
                    except (ValueError, TypeError):
                        pass
                for keyword in node.keywords:
                    if keyword.arg == "r":
                        try:
                            length = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            pass
                result_name = None
                parent = node
                while parent in parents:
                    parent = parents[parent]
                    if isinstance(parent, ast.Assign):
                        result_name = next(
                            (target.id for target in parent.targets if isinstance(target, ast.Name)),
                            None,
                        )
                        break
                permutation_calls.append(
                    (node.lineno, source_name, len(data_values[source_name]), length, result_name)
                )

    if not permutation_calls:
        return None

    permutation_calls.sort(key=lambda call: call[0])
    source_name = permutation_calls[0][1]
    item_count = permutation_calls[0][2]
    source_values = data_values[source_name]
    detail = (
        f"The script imports permutations from itertools and sets {source_name} "
        f"to {source_values}. It creates ordered selections from these {item_count} "
        "items. Each selection uses an input position at most once, and changing the "
        "order creates a different permutation."
    )

    for _, _, count, length, result_name in permutation_calls:
        if length is None:
            number = math.perm(count, count)
            detail += (
                f"\n\npermutations({source_name}) uses all {count} items and produces "
                f"{number} full-length permutations."
            )
        elif isinstance(length, int) and 0 <= length <= count:
            number = math.perm(count, length)
            detail += (
                f"\n\npermutations({source_name}, r={length}) produces {number} "
                f"ordered selections of length {length}. The r argument sets the "
                "length of each result."
            )
        if result_name:
            detail += f" The tuples are collected in {result_name}."
        else:
            continue

    detail += " Each result is a tuple, and list(...) collects the tuples in a list."
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        for node in ast.walk(tree)
    ):
        detail += " The script prints the collected lists."
    return detail


def explain_two_sum(code: str) -> str | None:
    """Give a useful walkthrough for the common one-pass Two Sum algorithm."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for child in ast.walk(node):
            if not isinstance(child, (ast.For, ast.AsyncFor)):
                continue
            iterator = child.iter
            if not (
                isinstance(iterator, ast.Call)
                and isinstance(iterator.func, ast.Name)
                and iterator.func.id == "enumerate"
                and iterator.args
                and isinstance(iterator.args[0], ast.Name)
            ):
                continue

            list_name = iterator.args[0].id
            pair_return = False
            complement_name = None
            dictionary_name = None
            for inner in ast.walk(child):
                if isinstance(inner, ast.If) and isinstance(inner.test, ast.Compare):
                    test = inner.test
                    if (
                        isinstance(test.left, ast.Name)
                        and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.In)
                        and test.comparators
                        and isinstance(test.comparators[0], ast.Name)
                    ):
                        complement_name = test.left.id
                        dictionary_name = test.comparators[0].id
                        pair_return = any(
                            isinstance(candidate, ast.Return)
                            for candidate in ast.walk(inner)
                        )
                        if pair_return:
                            break

            if pair_return and complement_name and dictionary_name:
                return (
                    f"The {node.name} method searches {list_name} for two values whose sum equals the target. "
                    f"It scans the list once and stores each number and its index in {dictionary_name}. "
                    f"For every number, it calculates the needed partner ({complement_name}) by subtracting "
                    "the current number from the target. If that partner is already in the dictionary, "
                    "the method returns the partner's earlier index and the current index. Otherwise, it "
                    f"records the current number and continues. This takes O(n) time and O(n) extra space. "
                    "If no matching pair is found, the method reaches the end and returns None implicitly."
                )
    return None


def explain_python_program(code: str) -> str | None:
    """Create a plain-language outline from the structure of valid Python code."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError):
        return None

    def describe(statements, depth=0):
        lines = []
        indent = "  " * depth
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in statement.args.posonlyargs + statement.args.args + statement.args.kwonlyargs]
                if statement.args.vararg:
                    args.append("*" + statement.args.vararg.arg)
                if statement.args.kwarg:
                    args.append("**" + statement.args.kwarg.arg)
                kind = "asynchronous function" if isinstance(statement, ast.AsyncFunctionDef) else "function"
                lines.append(f"{indent}Defines the {kind} {statement.name}({', '.join(args)}). Its body runs when the function is called.")
                lines.extend(describe(statement.body, depth + 1))
            elif isinstance(statement, ast.ClassDef):
                lines.append(f"{indent}Defines the class {statement.name}.")
                lines.extend(describe(statement.body, depth + 1))
            elif isinstance(statement, ast.If):
                lines.append(f"{indent}Checks whether {ast.unparse(statement.test)} is true.")
                lines.append(f"{indent}If true:")
                lines.extend(describe(statement.body, depth + 1) or [f"{indent}  No statements."])
                if statement.orelse:
                    label = "Otherwise, if the condition is false:" if not isinstance(statement.orelse[0], ast.If) else "Otherwise:"
                    lines.append(f"{indent}{label}")
                    lines.extend(describe(statement.orelse, depth + 1))
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                lines.append(f"{indent}Loops over {ast.unparse(statement.iter)}, assigning each item to {ast.unparse(statement.target)}.")
                lines.extend(describe(statement.body, depth + 1) or [f"{indent}  No statements."])
                if statement.orelse:
                    lines.append(f"{indent}After the loop finishes without a break:")
                    lines.extend(describe(statement.orelse, depth + 1))
            elif isinstance(statement, ast.While):
                lines.append(f"{indent}Repeats while {ast.unparse(statement.test)} is true.")
                lines.extend(describe(statement.body, depth + 1) or [f"{indent}  No statements."])
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = getattr(statement, "targets", None) or [statement.target]
                names = ", ".join(ast.unparse(target) for target in targets)
                lines.append(f"{indent}Sets {names} to {ast.unparse(statement.value)}.")
            elif isinstance(statement, ast.AugAssign):
                lines.append(f"{indent}Updates {ast.unparse(statement.target)} using {ast.unparse(statement.op)} and {ast.unparse(statement.value)}.")
            elif isinstance(statement, ast.Return):
                value = ast.unparse(statement.value) if statement.value else "nothing"
                lines.append(f"{indent}Returns {value} and exits the current function.")
            elif isinstance(statement, ast.Expr):
                value = statement.value
                if isinstance(value, ast.Call):
                    name = ast.unparse(value.func)
                    if name == "print":
                        content = ", ".join(ast.unparse(arg) for arg in value.args) or "an empty line"
                        lines.append(f"{indent}Prints {content}.")
                    else:
                        lines.append(f"{indent}Calls {ast.unparse(value)}.")
                elif isinstance(value, (ast.Constant, ast.Str)) and isinstance(getattr(value, "value", None), str):
                    continue  # Module and function docstrings do not perform an action.
                else:
                    lines.append(f"{indent}Evaluates {ast.unparse(value)}.")
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                lines.append(f"{indent}Imports {ast.unparse(statement)}.")
            elif isinstance(statement, ast.Try):
                lines.append(f"{indent}Runs a guarded block and handles matching exceptions.")
                lines.extend(describe(statement.body, depth + 1))
                for handler in statement.handlers:
                    label = ast.unparse(handler.type) if handler.type else "any exception"
                    lines.append(f"{indent}If {label} is raised:")
                    lines.extend(describe(handler.body, depth + 1))
                if statement.orelse:
                    lines.append(f"{indent}If no exception is raised:")
                    lines.extend(describe(statement.orelse, depth + 1))
                if statement.finalbody:
                    lines.append(f"{indent}Finally:")
                    lines.extend(describe(statement.finalbody, depth + 1))
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                lines.append(f"{indent}Enters the context {ast.unparse(statement.items[0].context_expr)} and runs:")
                lines.extend(describe(statement.body, depth + 1))
            elif isinstance(statement, ast.Raise):
                lines.append(f"{indent}Raises {ast.unparse(statement.exc) if statement.exc else 'the current exception'}.")
            elif isinstance(statement, ast.Assert):
                lines.append(f"{indent}Checks that {ast.unparse(statement.test)} is true; otherwise it raises an assertion error.")
            elif isinstance(statement, ast.Match):
                lines.append(f"{indent}Compares {ast.unparse(statement.subject)} with the following patterns:")
                for case in statement.cases:
                    guard = f" when {ast.unparse(case.guard)}" if case.guard else ""
                    lines.append(f"{indent}  If it matches {ast.unparse(case.pattern)}{guard}:")
                    lines.extend(describe(case.body, depth + 2))
            elif isinstance(statement, ast.Break):
                lines.append(f"{indent}Exits the nearest loop.")
            elif isinstance(statement, ast.Continue):
                lines.append(f"{indent}Skips to the next loop iteration.")
            elif isinstance(statement, ast.Pass):
                lines.append(f"{indent}Does nothing at this point.")
            else:
                # Keep valid but less common Python statements visible in the outline.
                lines.append(f"{indent}Runs {ast.unparse(statement)}.")
        return lines

    lines = describe(tree.body)
    if not lines:
        return "The code contains no executable statements."
    return "Here is what the Python code does:\n\n" + "\n".join(lines)


def build_flowchart(code: str) -> dict:
    """Convert Python statements to a safe, display-ready execution outline."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError):
        return {"steps": [], "error": "Fix the Python syntax to see its execution flow."}

    def make_steps(statements):
        steps = []
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                steps.append({"type": "step", "label": f"Import {ast.unparse(statement)}"})
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = getattr(statement, "targets", None) or [statement.target]
                target_text = ", ".join(ast.unparse(target) for target in targets)
                steps.append({
                    "type": "step",
                    "label": f"Set {target_text} to {ast.unparse(statement.value)}",
                })
            elif isinstance(statement, ast.AugAssign):
                operator = {
                    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
                }.get(type(statement.op), "an operation on")
                steps.append({
                    "type": "step",
                    "label": f"Update {ast.unparse(statement.target)} with {operator} {ast.unparse(statement.value)}",
                })
            elif isinstance(statement, ast.Expr):
                value = statement.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print":
                    printed = ", ".join(ast.unparse(arg) for arg in value.args) or "an empty line"
                    label = f"Print {printed}"
                else:
                    label = f"Evaluate {ast.unparse(value)}"
                steps.append({"type": "step", "label": label})
            elif isinstance(statement, ast.If):
                branches = [{
                    "label": "Yes",
                    "steps": make_steps(statement.body) or [{"type": "step", "label": "Continue"}],
                }]
                branches.append({
                    "label": "No",
                    "steps": make_steps(statement.orelse) or [{"type": "step", "label": "Continue"}],
                })
                steps.append({"type": "decision", "label": f"Is {ast.unparse(statement.test)} true?", "branches": branches})
            elif isinstance(statement, ast.ClassDef):
                steps.append({
                    "type": "class",
                    "label": f"Define class {statement.name}",
                    "steps": make_steps(statement.body),
                })
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                steps.append({
                    "type": "loop",
                    "label": f"For each {ast.unparse(statement.target)} in {ast.unparse(statement.iter)}",
                    "steps": make_steps(statement.body),
                })
            elif isinstance(statement, ast.While):
                steps.append({
                    "type": "loop",
                    "label": f"While {ast.unparse(statement.test)} is true",
                    "steps": make_steps(statement.body),
                })
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = ", ".join(arg.arg for arg in statement.args.args)
                steps.append({
                    "type": "function",
                    "label": f"Define {statement.name}({arguments})",
                    "steps": [
                        *make_steps(statement.body),
                        {"type": "implicit", "label": "If no return runs, Python returns None"},
                    ],
                })
            elif isinstance(statement, ast.Return):
                return_value = ast.unparse(statement.value) if statement.value else "nothing"
                steps.append({"type": "return", "label": f"Return {return_value}"})
            elif isinstance(statement, ast.Break):
                steps.append({"type": "step", "label": "Exit the current loop"})
            elif isinstance(statement, ast.Continue):
                steps.append({"type": "step", "label": "Continue with the next loop item"})
            elif isinstance(statement, ast.Pass):
                steps.append({"type": "step", "label": "Do nothing"})
            elif isinstance(statement, ast.Try):
                branches = [{"label": "Try", "steps": make_steps(statement.body)}]
                branches.extend({
                    "label": f"If {ast.unparse(handler.type) if handler.type else 'an exception'} occurs",
                    "steps": make_steps(handler.body),
                } for handler in statement.handlers)
                if statement.orelse:
                    branches.append({"label": "If no exception occurs", "steps": make_steps(statement.orelse)})
                steps.append({"type": "decision", "label": "Run guarded code", "branches": branches})
                if statement.finalbody:
                    steps.extend(make_steps(statement.finalbody))
        return steps

    return {
        "steps": [
            {"type": "start", "label": "Start"},
            *make_steps(tree.body),
            {"type": "end", "label": "End"},
        ],
        "error": None,
    }

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully.")


@app.post("/explain")
async def explain_code(request: CodeRequest):
    code = request.code.strip()
    if not code:
        raise HTTPException(status_code=422, detail="Code cannot be blank.")

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    code_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    explanation = (
        explain_two_sum(code)
        or explain_permutations(code)
        or explain_arithmetic_return(code)
        or explain_python_program(code)
    )
    outputs = None
    if explanation is None:
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="The model could not explain this code.") from exc

        generated_ids = outputs.sequences[0]
        explanation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Generation returns one cross-attention tensor per decoding step, with
    # shape (batch, heads, 1, source_tokens). Average heads and preserve the
    # step dimension so each returned row corresponds to a generated token.
    attention_rows = []
    if outputs is not None and outputs.cross_attentions:
        for step in outputs.cross_attentions:
            if not step:
                continue
            last_layer_attention = step[-1][0, :, 0, :]
            attention_rows.append(last_layer_attention.mean(dim=0))

    if outputs is None:
        explanation_tokens = tokenizer.tokenize(explanation)
        attention_map = []
    elif attention_rows:
        all_attention = torch.stack(attention_rows)
        # The decoder start token is present in sequences but has no generation
        # attention row. Pair the remaining generated IDs and attention rows,
        # dropping EOS and any other special tokens from both.
        generated_token_ids = generated_ids[1 : 1 + len(attention_rows)]
        visible = [
            (token_id, row)
            for token_id, row in zip(generated_token_ids.tolist(), all_attention)
            if token_id not in tokenizer.all_special_ids
        ]
        explanation_tokens = tokenizer.convert_ids_to_tokens(
            [token_id for token_id, _ in visible]
        )
        attention_map = [row.tolist() for _, row in visible]
    else:
        explanation_tokens = tokenizer.convert_ids_to_tokens(
            [token_id for token_id in generated_ids.tolist() if token_id not in tokenizer.all_special_ids]
        )
        attention_map = []

    return {
        "explanation": explanation,
        "explanation_tokens": explanation_tokens,
        "code_tokens": code_tokens,
        "attention_map": attention_map,
        "flowchart": build_flowchart(code),
    }
