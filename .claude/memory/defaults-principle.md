# Defaults Principle

**THE DEFAULT SHOULD BE SET ONLY IN THE PARAMETERS (Click options).**

## Correct Pattern

```python
@click.option(
    "--lambda",
    "lambda_val",
    type=float,
    default=DEFAULT_LAMBDA,  # Default set HERE ONLY
    show_default=True,
    callback=validate_positive,
    help="Lambda parameter for infection times",
)
def run_command(lambda_val, ...):
    # Use lambda_val directly - it already has the default or user's value
    do_something(lambda_val)
```

## Wrong Pattern

```python
# WRONG - spreading defaults throughout the code
def some_function(lambda_val):
    value = lambda_val or DEFAULT_LAMBDA  # NO!
    value = something.get("lambda", DEFAULT_LAMBDA)  # NO!
```

## Key Points

1. `DEFAULT_*` constants should ONLY be used in `@click.option` decorators
2. Everywhere else uses the parameter value that Click provides
3. If user passes a value → use that value
4. If user doesn't pass a value → Click already applied the default
5. NO fallbacks, NO spreading defaults across the code
6. For experiment JSON files: require explicit values, fail with clear error if missing
