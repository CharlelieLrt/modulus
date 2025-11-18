<!-- markdownlint-disable MD012 MD013 MD024 MD031 MD032 MD033 MD034 MD040 MD046 -->
<!-- MD012: Multiple consecutive blank lines -->
<!-- MD013: Line length -->
<!-- MD024: Multiple headings with the same content -->
<!-- MD031: Fenced code blocks should be surrounded by blank lines -->
<!-- MD032: Lists should be surrounded by blank lines -->
<!-- MD033: Inline HTML -->
<!-- MD034: Bare URL used -->
<!-- MD040: Fenced code blocks should have a language specified -->
<!-- MD046: Code block style -->

# MODELS_IMPLEMENTATION - Coding Standards

## Overview

This document defines the coding standards and best practices for implementing
model classes in the PhysicsNeMo repository. These rules are designed to ensure
consistency, maintainability, and high code quality across all model
implementations.

**Important:** These rules are enforced as strictly as possible. Deviations
from these standards should only be made when absolutely necessary and must be
documented with clear justification in code comments and approved during code
review.

## Document Organization

This document is structured in two main sections:

1. **Rule Index**: A quick-reference table listing all rules with their IDs,
   one-line summaries, and the context in which they apply. Use this section
   to quickly identify relevant rules when implementing or reviewing code.

2. **Detailed Rules**: Comprehensive descriptions of each rule, including:
   - Clear descriptions of what the rule requires
   - Rationale explaining why the rule exists
   - Examples demonstrating correct implementation
   - Anti-patterns showing common mistakes to avoid

## How to Use This Document

- **When creating new models**: Review all rules before starting implementation,
  paying special attention to rules MOD-000 through MOD-003.
- **When reviewing code**: Use the Rule Index to quickly verify compliance with
  all applicable rules.
- **When refactoring**: Ensure refactored code maintains or improves compliance
  with these standards.
- **For AI agents that generate code**: This document is formatted for easy parsing. Each rule has
  a unique ID and structured sections (Description, Rationale, Example,
  Anti-pattern) that can be extracted and used as context. When generating code
  based on a rule, an AI agent should explicitly quote the rule ID that it is
  following, and explicitly quote the relevant extract from the rule that it is
  using as context. For example, "Following rule MOD-000, the new model class
  should be ..."
- **For AI agents that review code**: When reviewing code, the AI agent should
  explicitly identify which rules are violated by the code, and provide a clear
  explanation of why the code violates the rule. The AI agent should explicitly
  quote the rule ID that the code is violating, and explicitly quote the relevant
  extract from the rule that it is using as context. For example, "Code violates
  rule MOD-000, because the new model class is not..."

## Rule Index

| Rule ID | Summary | Apply When |
|---------|---------|------------|
| [`MOD-000`](#mod-000-models-organization-and-where-to-place-new-model-classes) | Models organization and where to place new model classes | Creating or refactoring new model classes |
| [`MOD-001`](#mod-001-use-proper-class-inheritance-for-all-models) | Use proper class inheritance for all models | Creating or refactoring new model classes |
| [`MOD-002`](#mod-002-model-classes-lifecycle) | Model classes lifecycle | Creating or moving existing model classes |
| [`MOD-003`](#mod-003-model-classes-documentation) | Model classes documentation | Creating or editing any docstring in a model class |
| [`MOD-004`](#mod-004-self-contained-model-modules) | Keep utility functions in the same module as the model | Organizing or refactoring model code |
| [`MOD-005`](#mod-005-tensor-shape-validation) | Validate tensor shapes in forward and public methods | Implementing or modifying model forward or public methods |
| [`MOD-006`](#mod-006-jaxtyping-annotations) | Use jaxtyping for tensor type annotations | Adding or editing any new public method of a model class |
| [`MOD-007`](#mod-007-backward-compatibility) | Maintain backward compatibility for model signatures | Modifying existing production models |
| [`MOD-008`](#mod-008-minimal-ci-testing-requirements) | Provide comprehensive CI tests for all models | Moving models out of experimental or adding new models |

---

## Detailed Rules

### MOD-000: Models organization and where to place new model classes

**Description:**

There are two types of models in PhysicsNeMo:

- Reusable layers that are the building blocks of more complex architectures.
  Those should go into `physicsnemo/nn`. Those include for instance
  `FullyConnected`, various variants of attention layers, `UNetBlock` (a block
  of a U-Net), etc.
  All layers that are directly exposed to the user should be imported in
  `physicsnemo/nn/__init__.py`, such that they can be used as follows:
  ```python
  from physicsnemo.nn import MyLayer
  ```
- More complete models, composed of multiple layers and/or other sub-models.
  Those should go into `physicsnemo/models`. All models that are directly
  exposed to the user should be imported in `physicsnemo/models/__init__.py`,
  such that they can be used as follows:
  ```python
  from physicsnemo.models import MyModel
  ```

The only exception to this rule is for models or layers that are highly specific to a
single example. In this case, it may be acceptable to place them in a module
specific to the example code, such as for example
`examples/<example_name>/utils/nn.py`.

**Rationale:**
Ensures consistency and clarity in the organization of models in the
repository, in particular a clear separation between reusable layers and more
complete models that are applicable to a specific domain or specific data
modality.

**Example:**

```python
# Good: Reusable layer in physicsnemo/nn/attention.py
class MultiHeadAttention(Module):
    """A reusable attention layer that can be used in various architectures."""
    pass

# Good: Complete model in physicsnemo/models/transformer.py
class TransformerModel(Module):
    """A complete transformer model composed of attention and feedforward layers."""
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(...)
        self.ffn = FeedForward(...)

# Good: Example-specific utility in examples/weather/utils/nn.py
class WeatherSpecificLayer(Module):
    """Layer highly specific to the weather forecasting example."""
    pass
```

**Anti-pattern:**

```python
# WRONG: Complete model placed in physicsnemo/nn/ instead of physicsnemo/models/
# File: physicsnemo/nn/transformer.py
class TransformerModel(Module):
    """Should be in physicsnemo/models/ not physicsnemo/nn/"""
    pass

# WRONG: Reusable layer placed in physicsnemo/models/ instead of physicsnemo/nn/
# File: physicsnemo/models/attention.py
class MultiHeadAttention(Module):
    """Should be in physicsnemo/nn/ not physicsnemo/models/"""
    pass
```

---

### MOD-001: Use proper class inheritance for all models

**Description:**
All model classes must inherit from `physicsnemo.Module`. Direct subclasses of
`torch.nn.Module` are not allowed. Direct subclasses of `physicsnemo.Module`
are allowed (note that `physicsnemo.Module` is a subclass of `torch.nn.Module`).
Ensure proper initialization of parent classes using `super().__init__()`. Pass
the `meta` argument to the `super().__init__()` call if appropriate, otherwise
set it manually with `self.meta = meta`.

**Rationale:**
Ensures invariants and functionality of the `physicsnemo.Module` class for all
models. In particular, instances of `physicsnemo.Module` benefit from features
that are not available in `torch.nn.Module` instances. Those include serialization
for checkpointing and loading modules and submodules, versioning system to
handle backward compatibility, as well as ability to be registered in the
`physicsnemo.registry` for easy instantiation and use in any codebase.

**Example:**

```python
from physicsnemo import Module

class MyModel(Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(meta=MyModelMetaData())
        self.linear = nn.Linear(input_dim, output_dim)
```

**Anti-pattern:**

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        self.linear = nn.Linear(input_dim, output_dim)
```

---

### MOD-002: Model classes lifecycle

**Description:**
All model classes must follow the following lifecycle:

- Stage 1: Creation. This is the stage where the model class is created. For
  the vast majority of models, new classes are created either in
  `physicsnemo/experimental/nn` for reusable layers, or in
  `physicsnemo/experimental/models` for more complete models. The `experimental`
  folder is used to store models that are still under development (beta or
  alpha releases) during this stage, backward compatibility is not guaranteed.
  One exception is when the developer is highly confident that the model
  is sufficiently mature and applicable to many domains or use cases. In this
  case the model class can be created in the `physicsnemo/nn` or `physicsnemo/models`
  folders directly, and backward compatibility is guaranteed. Another exception
  is when the model class is highly specific to a single example. In this case,
  it may be acceptable to place it in a module specific to the example code,
  such as for example `examples/<example_name>/utils/nn.py`.

- Stage 2: Production. After staying in stage 1 for a sufficient amount of time
  (typically at least 1 release cycle), the model class is promoted to stage 2.
  It is then moved to the `physicsnemo/nn` or `physicsnemo/models` folders,
  based on the rule `MOD-000`. During this stage, backward compatibility is
  guaranteed.

- Stage 3: Pre-deprecation. For a model class in stage 3 in `physicsnemo/nn` or
  `physicsnemo/models`, the developer should start planning its deprecation.
  This is done by adding a warning message to the model class, indicating that
  the model class is deprecated and will be removed in a future release. The
  warning message should be a clear and concise message that explains why the
  model class is being deprecated and what the user should do instead. The
  deprecation message should be added to both the docstring and should be
  raised at runtime. The developer is free to choose the mechanism to raise the
  deprecation warning. A model class cannot be deprecated without staying in
  stage 3 "pre-deprecation" for at least 1 release cycle.

- Stage 4: Deprecation. After staying in stage 3 "pre-deprecation" for at least 1
  release cycle, the model class is deprecated. It can be deleted from the
  codebase.

**Rationale:**
This lifecycle ensures a structured approach to model development and maintenance.
The experimental stage allows rapid iteration without backward compatibility
constraints, enabling developers to refine APIs based on user feedback. The
production stage provides stability for users who depend on these models. The
pre-deprecation and deprecation stages ensure users have sufficient time to
migrate to newer alternatives, preventing breaking changes that could disrupt
their workflows. This graduated approach balances innovation with stability,
a critical requirement for a scientific computing framework.

**Example:**

```python
# Good: Stage 1 - New experimental model
# File: physicsnemo/experimental/models/new_diffusion.py
class DiffusionModel(Module):
    """New diffusion model under active development. API may change."""
    pass

# Good: Stage 2 - Promoted to production after 1 release cycle
# File: physicsnemo/models/diffusion.py (moved from experimental/)
class DiffusionModel(Module):
    """Stable diffusion model with backward compatibility guarantees."""
    pass

# Good: Stage 3 - Pre-deprecation with warning
# File: physicsnemo/models/old_diffusion.py
class DiffusionModel(Module):
    """
    Legacy diffusion model.

    .. deprecated:: 0.5.0
        ``OldDiffusionModel`` is deprecated and will be removed in version 0.7.0.
        Use :class:`~physicsnemo.models.NewDiffusionModel` instead.
    """
    def __init__(self):
        import warnings
        warnings.warn(
            "OldDiffusionModel is deprecated. Use DiffusionModel instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()

# Good: Stage 4 - Model removed after deprecation period
# (File deleted from codebase)
```

**Anti-pattern:**

```python
# WRONG: New model directly in production folder without experimental phase
# File: physicsnemo/models/brand_new_model.py (should be in experimental/ first)
class BrandNewModel(Module):
    """Skipped experimental stage - risky for stability"""
    pass

# WRONG: Breaking changes in production without deprecation cycle
# File: physicsnemo/models/diffusion.py
class DiffusionModel(Module):
    def __init__(self, new_required_param):  # Breaking change!
        # Changed API without deprecation warning - breaks user code
        pass

# WRONG: Deprecation without sufficient warning period
# (Model deprecated and removed in same release)

# WRONG: No deprecation warning in code
# File: physicsnemo/models/old_model.py
class OldModel(Module):
    """Will be removed next release."""  # Docstring mentions it but no runtime warning
    def __init__(self):
        # Missing: warnings.warn(..., DeprecationWarning)
        super().__init__()
```

---

### MOD-003: Model classes documentation

**Description:**

Every new model or modification of any model code should be documented with a
comprehensive docstring. Each method of the model class should be documented with a
docstring as well. The forward method should be documented in the docstring of the
model class, instead of being in the docstring of the forward method itself. A
docstring for the forward is still possible but it should be concise and to the
point. To document the forward method, use the sections `Forward` and
`Outputs`. In addition, all docstrings should be written in the NumPy style,
and adopt formatting to be compatible with our Sphinx restructured text (RST)
documentation.
The docstrings should follow the following requirements:

- Each docstring should be prefixed with `r"""`.

- The class docstring should at least contain three sections: `Parameters`,
  `Forward`, and `Outputs`. Other sections such as `Notes`, `Examples`,
  or `..important::` or `..code-block:: python` are possible. Other sections
  are not recognized by our Sphinx documentation and are prohibited.

- All methods should be documented with a docstring, with at least a `Parameters`
  section and a `Returns` section. Other sections such as `Notes`, `Examples`,
  or `..important::` or `..code-block:: python` are possible. Other sections
  are not recognized by our Sphinx documentation and are prohibited.

- All tensors should be documented with their shape, using LaTeX math notation such
  as :math:`(N, C, H_{in}, W_{in})` (there is flexibility for naming the
  dimensions, but the math format should be enforced). Our documentation is
  rendered using LaTeX, and supports a rich set of LaTeX commands, so
  it is recommended to use LaTeX commands whenever possible for mathematical
  variables in the docstrings. The mathematical notations should be to some degree
  consistent with the actual variable names in the code (even though
  that is not always possible, to avoid too complex formatting).

- For arguments or variables that are callback functions, (e.g. Callable), the
  docstring should include a clear separated ..code-block:: that specifies the
  required signature and return type of the callback function. This is not only
  true for callback functions, but for any type of parameters or arguments that
  has some complex type specification or API requirements. The explanation code
  block should be placed in the top or bottom section of the docstrings, but
  not in the `Parameters` or `Forward` or `Outputs` sections, for readability
  and clarity.

- Inline code should be formatted double backticks, such as ``my_variable``.
  Single backticks are not allowed as they don't render properly in our Sphinx
  documentation.

- All parameters should be documented with their type and default values on a
  single line.

- When possible, docstrings should use links to other docstrings, such as
  :class:`~physicsnemo.models.some_model.SomeModel`, or
  :func:`~physicsnemo.utils.common_function`, or
  :meth:`~physicsnemo.models.some_model.SomeModel.some_method`.

- When referencing external resources, such as papers, websites, or other
  documentation, docstrings should use links to the external resource in the
  format `some link text <some_url>`_.

- Docstrings are strongly encouraged to have an `Examples` section that
  demonstrates basic construction and usage of the model. These example sections
  serve as both documentation and tests, as our CI system automatically tests
  these code sections for correctness when present. Examples should be
  executable Python code showing typical use cases, including model
  instantiation, input preparation, and forward pass execution.

**Rationale:**
Comprehensive and well-formatted documentation is essential for scientific
software. It enables users to understand model capabilities, expected inputs,
and outputs without inspecting source code. LaTeX math notation and proper
formatting ensure documentation renders correctly in Sphinx, creating
professional, publication-quality documentation. Consistent documentation
standards facilitate automatic documentation generation, improve code
discoverability, and help AI agents understand code context. For a framework
used in scientific research, clear documentation of tensor shapes, mathematical
formulations, and API contracts is critical for reproducibility and correct usage.

**Example:**

```python
from typing import Callable, Optional
from physicsnemo.models import Module
import torch

class SimpleEncoder(Module):
    r"""
    # Rule: Docstring starts with r (for raw string) followed by three double quotes for proper LaTeX rendering
    A simple encoder network that transforms input features to a latent representation.

    This model applies a sequence of linear transformations with activation functions
    to encode input data into a lower-dimensional latent space. The architecture is
    based on the approach described in `Autoencoder Networks <https://arxiv.org/example>`_.
    # Rule: External references use proper link format

    The model supports custom preprocessing via a callback function that is applied
    to the input before encoding.

    .. code-block:: python

        # Rule: Callback functions or complex API requirements documented in a concise code-block
        def preprocess_fn(x: torch.Tensor) -> torch.Tensor:
            ...
            return y
    
    where ``x`` is the input tensor of shape :math:`(B, D_{in})` and ``y`` is the output tensor of shape :math:`(B, D_{out})`.

    Parameters
    ----------
    # Rule: Parameters section is mandatory
    input_dim : int
        # Rule: Type and description on single line
        Dimension of input features.
    latent_dim : int
        Dimension of the latent representation.
    hidden_dim : int, optional, default=128
        # Rule: Default values are documented on same line
        Dimension of the hidden layer.
    activation : str, optional, default="relu"
        Activation function to use. See :func:`~torch.nn.functional.relu` for details.
        # Rule: Cross-references use proper Sphinx syntax
    preprocess_fn : Callable[[torch.Tensor], torch.Tensor], optional, default=None
        Optional preprocessing function applied to input. See the code block above
        for the required signature.
        # Rule: Callback functions documented with reference to code-block

    Forward
    # Rule: Forward section documents the forward method in class docstring
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})` where :math:`B` is batch size
        # Rule: Tensor shapes use LaTeX math notation with :math:
        and :math:`D_{in}` is input dimension.
    return_hidden : bool, optional, default=False
        If ``True``, also returns hidden layer activations.
        # Rule: Inline code uses double backticks

    Outputs
    # Rule: Outputs section is mandatory
    -------
    torch.Tensor or tuple
        If ``return_hidden`` is ``False``, returns latent representation of shape
        :math:`(B, D_{latent})`. If ``True``, returns tuple of
        (latent, hidden) where hidden has shape :math:`(B, D_{hidden})`.

    Examples
    # Rule: Examples section is allowed and helpful
    --------
    >>> model = SimpleEncoder(input_dim=784, latent_dim=64)
    >>> x = torch.randn(32, 784)
    >>> latent = model(x)
    >>> latent.shape
    torch.Size([32, 64])

    Notes
    # Rule: Notes section is allowed for additional context
    -----
        This encoder can be used as part of a larger autoencoder architecture
        by combining it with a decoder network such as
        :class:`~physicsnemo.models.decoder.SimpleDecoder`.
        # Rule: Cross-references to other classes use proper Sphinx syntax
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        activation: str = "relu",
        preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        super().__init__(meta=SimpleEncoderMetaData())
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, latent_dim)
        self.preprocess_fn = preprocess_fn

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """Concise forward docstring referencing class docstring for details."""
        # Rule: Forward method can have concise docstring, main docs in class
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        hidden = torch.relu(self.fc1(x))
        latent = self.fc2(hidden)
        if return_hidden:
            return latent, hidden
        return latent

    def compute_reconstruction_loss(
        self,
        latent: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute mean squared error between latent representation and target.
        # Rule: Methods have their own docstrings with Parameters and Returns sections

        Parameters
        ----------
        # Rule: Parameters section is mandatory for methods
        latent : torch.Tensor
            Latent representation of shape :math:`(B, D_{latent})`.
            # Rule: Tensor shapes documented with :math:
        target : torch.Tensor
            Target tensor of shape :math:`(B, D_{latent})`.

        Returns
        -------
        # Rule: Returns section is mandatory for methods (not "Outputs" - that's for forward)
        torch.Tensor
            Scalar loss value.
        """
        return torch.nn.functional.mse_loss(latent, target)
```

**Anti-pattern:**

```python
from physicsnemo.models import Module
import torch

class BadEncoder(Module):
    '''
    # WRONG: Should use r (for raw string) followed by three double quotes for docstrings not three single quotes
    A simple encoder network
    # WRONG: missing Parameters, Forward, or Outputs sections in the docstring
    # WRONG: callback function preprocess_fn is not documented at all
    '''
    def __init__(self, input_dim, latent_dim, hidden_dim=128, preprocess_fn=None):
        # WRONG: No type hints in signature
        # WRONG: preprocess_fn callback parameter is missing from docstring entirely
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, latent_dim)
        self.preprocess_fn = preprocess_fn

    def forward(self, x, return_hidden=False):
        # WRONG: No type hints
        """
        Forward pass of the encoder.
        # WRONG: Should be brief, main docs for the forward method should be in the class docstring

        Args:
            x: input tensor with shape (B, D_in)
            # WRONG: Using "Args" instead of NumPy-style "Parameters"
            # WRONG: Not using :math: with backticks for shapes and other mathematical notations
            return_hidden (bool): whether to return hidden state
            # WRONG: Mixing documentation styles

        Returns:
            encoded representation
            # WRONG: Using "Returns" instead of "Outputs". The "Returns" section is for methods other than the forward method. The forward method should have an "Outputs" section instead.
            # WRONG: No shape information
        """
        # WRONG: Entire forward signature and behavior should be in class docstring
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        hidden = torch.relu(self.fc1(x))
        latent = self.fc2(hidden)
        if return_hidden:
            return latent, hidden
        return latent

    def compute_loss(self, x, y):
        # WRONG: No type hints
        """
        Compute the loss
        # WRONG: No proper docstring structure

        Description:
            # WRONG: "Description" is not a recognized section, should be in the main description
            This method computes some loss value

        Arguments:
            # WRONG: "Arguments" is not recognized, should be "Parameters"
            x: first input (B, D)
            # WRONG: Not using :math: for tensor shapes
            y: second input
            # WRONG: No shape information

        Output:
            # WRONG: "Output" (singular) is not recognized, should be "Returns" for methods
            loss value
            # WRONG: No type or shape information
        """
        return torch.nn.functional.mse_loss(x, y)

    def helper_method(self, x):
        # WRONG: No docstring at all for method, docstrings for methods are mandatory
        # WRONG: missing Parameters and Returns sections in the docstring
        # WRONG: No type hints
        return x * 2
```

---

### MOD-004: Self-contained model modules

**Description:**

All utility functions for a model class should be contained in the same module
file as the model class itself. For a model called `MyModelName` in
`my_model_name.py`, all utility functions specific to that model should also be
in `my_model_name.py`. Utility functions should never be placed in separate
files like `my_model_name_utils.py` or `my_model_name/utils.py`.

The only exception to this rule is when a utility function is used across
multiple models. In that case, the shared utility should be placed in an
appropriate shared module and imported in `my_model_name.py`.

**Rationale:**

Self-contained modules are easier to understand, maintain, and navigate. Having
all model-specific code in one place reduces cognitive load and makes it clear
which utilities are model-specific versus shared. This also simplifies code
reviews and reduces the likelihood of orphaned utility files when models are
refactored or removed.

**Example:**

```python
# Good: Utility function in the same file as the model
# File: physicsnemo/models/my_transformer.py

def _compute_attention_mask(seq_length: int) -> torch.Tensor:
    """Helper function specific to MyTransformer."""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

class MyTransformer(Module):
    """A transformer model."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = _compute_attention_mask(x.shape[1])
        return self._apply_attention(x, mask)
```

**Anti-pattern:**

```python
# WRONG: Utility function in separate file
# File: physicsnemo/models/my_transformer_utils.py
def _compute_attention_mask(seq_length: int) -> torch.Tensor:
    """Should be in my_transformer.py, not in a separate utils file."""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

# File: physicsnemo/models/my_transformer.py
from physicsnemo.models.my_transformer_utils import _compute_attention_mask  # WRONG

class MyTransformer(Module):
    """A transformer model."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = _compute_attention_mask(x.shape[1])
        return self._apply_attention(x, mask)
```

---

### MOD-005: Tensor shape validation

**Description:**

All forward methods and other public methods that accept tensor arguments must
validate tensor shapes at the beginning of the method. This rule applies to:
- Individual tensor arguments
- Containers of tensors (lists, tuples, dictionaries)

For containers, validate their length, required keys, and the shapes of
contained tensors. Validation statements should be concise (ideally one check
per argument). Error messages must follow the standardized format:
`"Expected tensor of shape (B, D) but got tensor of shape {actual_shape}"`.

To avoid interactions with `torch.compile`, all validation must be wrapped in a
conditional check using `torch.compiler.is_compiling()`. Follow the "fail-fast"
approach by validating inputs before any computation.

**Rationale:**

Early shape validation catches errors at the API boundary with clear, actionable
error messages, making debugging significantly easier. Without validation, shape
mismatches result in cryptic errors deep in the computation graph. The
`torch.compile` guard ensures that validation overhead is eliminated in
production compiled code while preserving debug-time safety.

**Example:**

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward pass with shape validation."""
    ### Input validation
    # Skip validation when running under torch.compile for performance
    if not torch.compiler.is_compiling():
        # Extract expected dimensions
        B, C, H, W = x.shape if x.ndim == 4 else (None, None, None, None)

        # Validate x shape
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C} channels"
            )

        # Validate optional mask
        if mask is not None:
            if mask.shape != (B, H, W):
                raise ValueError(
                    f"Expected mask shape ({B}, {H}, {W}), got {tuple(mask.shape)}"
                )

    # Actual computation happens after validation
    return self._process(x, mask)

def process_list(self, tensors: List[torch.Tensor]) -> torch.Tensor:
    """Process a list of tensors with validation."""
    ### Input validation
    if not torch.compiler.is_compiling():
        if len(tensors) == 0:
            raise ValueError("Expected non-empty list of tensors")

        # Validate all tensors have consistent shapes
        ref_shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], start=1):
            if t.shape != ref_shape:
                raise ValueError(
                    f"All tensors must have the same shape. "
                    f"Tensor 0 has shape {tuple(ref_shape)}, "
                    f"but tensor {i} has shape {tuple(t.shape)}"
                )

    return torch.stack(tensors)
```

**Anti-pattern:**

```python
# WRONG: No validation at all
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layer(x)  # Will fail with cryptic error if shape is wrong

# WRONG: Validation not guarded by torch.compiler.is_compiling()
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:  # Breaks torch.compile
        raise ValueError(f"Expected 4D tensor, got {x.ndim}D")
    return self.layer(x)

# WRONG: Validation after computation has started
def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.layer1(x)  # Computation started
    if x.shape[1] != self.in_channels:  # Too late!
        raise ValueError(f"Wrong number of channels")
    return self.layer2(h)

# WRONG: Non-standard error message format
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not torch.compiler.is_compiling():
        if x.ndim != 4:
            raise ValueError("Input must be 4D")  # Missing actual shape info
    return self.layer(x)
```

---

### MOD-006: Jaxtyping annotations

**Description:**

All tensor arguments and variables in model `__init__`, `forward`, and other
public methods must have type annotations using `jaxtyping`. This provides
runtime-checkable shape information in type hints.

Use the format `Float[torch.Tensor, "shape_spec"]` where shape_spec describes
tensor dimensions using space-separated dimension names (e.g., `"batch channels height width"`
or `"b c h w"`).

**Rationale:**

Jaxtyping annotations provide explicit, machine-readable documentation of
expected tensor shapes. This enables better IDE support, catches shape errors
earlier, and makes code more self-documenting. The annotations serve as both
documentation and optional runtime checks when jaxtyping's validation is
enabled.

**Example:**

```python
from jaxtyping import Float
import torch

class MyConvNet(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels height width"]
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        """Process input with convolution."""
        return self.conv(x)

def process_attention(
    query: Float[torch.Tensor, "batch seq_len d_model"],
    key: Float[torch.Tensor, "batch seq_len d_model"],
    value: Float[torch.Tensor, "batch seq_len d_model"]
) -> Float[torch.Tensor, "batch seq_len d_model"]:
    """Compute attention with clear shape annotations."""
    pass
```

**Anti-pattern:**

```python
# WRONG: No jaxtyping annotations
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layer(x)

# WRONG: Using plain comments instead of jaxtyping
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch, channels, height, width)  # Use jaxtyping instead
    return self.layer(x)

# WRONG: Incomplete annotations (missing jaxtyping for tensor arguments)
def forward(
    self,
    x: Float[torch.Tensor, "b c h w"],
    mask: torch.Tensor  # Missing jaxtyping annotation
) -> Float[torch.Tensor, "b c h w"]:
    return self.layer(x, mask)
```

---

### MOD-007: Backward compatibility

**Description:**

For any model in `physicsnemo/nn` or `physicsnemo/models`, it is strictly
forbidden to change the signature of `__init__`, any public method, or any
public attribute without maintaining backward compatibility. This includes:
- Adding new required parameters
- Removing parameters
- Renaming parameters
- Changing parameter types
- Changing return types

If a signature change is absolutely necessary, the developer must:
1. Add a backward compatibility mapping in the model class
2. Increment the model version number
3. Maintain support for the old API for at least 2 release cycles
4. Add deprecation warnings for the old API

**Rationale:**

PhysicsNeMo is used in production environments and research code where
unexpected API changes can break critical workflows. Maintaining backward
compatibility ensures that users can upgrade to new versions without their code
breaking. Version numbers and compatibility mappings provide a clear migration
path when changes are necessary.

**Example:**

```python
from typing import Any, Dict, Optional

# Good: Adding optional parameter with default value (backward compatible)
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"
    __supported_model_checkpoint_version__ = {
        "1.0": "Loading checkpoint from version 1.0 (current is 2.0). Still supported."
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,  # New parameter with default
        new_feature: bool = False  # New parameter with default
    ):
        super().__init__(meta=MyModelMetaData())
        # ... implementation

# Good: Proper backward compatibility when parameter must be renamed
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"
    __supported_model_checkpoint_version__ = {
        "1.0": (
            "Loading MyModel checkpoint from version 1.0 (current is 2.0). "
            "Parameter 'hidden_dim' has been renamed to 'hidden_size'. "
            "Consider re-saving to upgrade to version 2.0."
        )
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map arguments from older versions to current version format."""
        # Call parent class method first
        args = super()._backward_compat_arg_mapper(version, args)

        if version == "1.0":
            # Map old parameter name to new name
            if "hidden_dim" in args:
                args["hidden_size"] = args.pop("hidden_dim")

            # Remove deprecated parameters that are no longer used
            if "legacy_param" in args:
                _ = args.pop("legacy_param")

        return args

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,  # New name (was 'hidden_dim' in v1.0)
    ):
        super().__init__(meta=MyModelMetaData())
        self.hidden_size = hidden_size
        # ... implementation
```

**Anti-pattern:**

```python
# WRONG: Changing parameter name without backward compatibility
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"
    # Missing: __supported_model_checkpoint_version__ and _backward_compat_arg_mapper

    def __init__(self, input_dim: int, hidden_size: int):  # Renamed from hidden_dim
        super().__init__(meta=MyModelMetaData())
        # WRONG: Old checkpoints with 'hidden_dim' will fail to load!

# WRONG: Adding required parameter without default
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"

    def __init__(self, input_dim: int, output_dim: int, new_param: int):  # No default!
        super().__init__(meta=MyModelMetaData())
        # WRONG: Old checkpoints without 'new_param' will fail to load!

# WRONG: Not incrementing version when making breaking changes
class MyModel(Module):
    __model_checkpoint_version__ = "1.0"  # Should be "2.0"!

    @classmethod
    def _backward_compat_arg_mapper(cls, version: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # WRONG: Making breaking changes but not updating version number
        if "hidden_dim" in args:
            args["hidden_size"] = args.pop("hidden_dim")
        return args

# WRONG: Not calling super() in _backward_compat_arg_mapper
class MyModel(Module):
    @classmethod
    def _backward_compat_arg_mapper(cls, version: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # WRONG: Missing super()._backward_compat_arg_mapper(version, args)
        if version == "1.0":
            if "hidden_dim" in args:
                args["hidden_size"] = args.pop("hidden_dim")
        return args

# WRONG: Changing return type without compatibility
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # WRONG: v1.0 returned single tensor, v2.0 returns tuple - breaks user code!
        return output, loss
```

---

### MOD-008: Minimal CI testing requirements

**Description:**

Every model in a module file `my_model_name.py` in `physicsnemo/nn` or
`physicsnemo/models` must have corresponding tests in
`test/models/test_<my_model_name>.py`. Tests should roughly follow a similar
template and, three types of tests are required:

1. **Constructor and attribute tests**: Verify model instantiation and all
   public attributes (excluding buffers and parameters).

2. **Non-regression test with reference data**: Instantiate a model, run
   forward pass, and compare outputs against reference data saved in a `.pth`
   file.

3. **Non-regression test from checkpoint**: Load a model from a checkpoint file
   (`.mdlus`) and verify outputs match reference data.

Additional requirements:
- All tests must use `pytest` parameterization syntax
- At least 2 configurations must be tested: one with all default arguments, one
  with non-default arguments. More variations specific to some relevant
  use-cases are also encouraged.
- Test tensors must have realistic shapes (e.g. no singleton dimensions) and
  should be as meaningful and representative of actual use cases as possible.
- All public methods must have the same non-regression tests as the forward
  method
- Simply checking output shapes is NOT sufficient - actual values must be
  compared
- **Critical:** Per MOD-002, it is forbidden to move a model out of the
  experimental stage/directory without these tests

**Rationale:**

Comprehensive tests ensure model correctness and prevent regressions as code
evolves. Non-regression tests with reference data catch subtle numerical changes
that could break reproducibility. Checkpoint tests verify serialization and
deserialization work correctly. Parameterized tests ensure models work across
different configurations. These tests are required before models can graduate
from experimental to production status.

**Example:**

```python
# Good: Following the test_layers_unet_block.py template
import pytest
import torch
from physicsnemo.models import MyModel

def _instantiate_model(cls, seed: int = 0, **kwargs):
    """Helper to create model with reproducible parameters."""
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"]
)
def test_my_model_non_regression(device, config):
    """Test model forward pass against reference output."""
    # Setup model configuration
    if config == "default":
        model = _instantiate_model(MyModel, input_dim=64, output_dim=32)
    else:
        model = _instantiate_model(
            MyModel,
            input_dim=64,
            output_dim=32,
            hidden_dim=256,
            dropout=0.1
        )

    model = model.to(device)

    # Test constructor and attributes
    assert model.input_dim == 64
    assert model.output_dim == 32
    if config == "custom":
        assert model.hidden_dim == 256
        assert model.dropout == 0.1

    # Load reference data (meaningful shapes, no singleton dimensions)
    data = torch.load(f"test/models/data/my_model_{config}_v1.0.pth")
    x = data["x"].to(device)  # Shape: (4, 64), not (1, 64)
    out_ref = data["out"].to(device)

    # Run forward and compare
    out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_my_model_from_checkpoint(device):
    """Test loading model from checkpoint and verify outputs."""
    model = physicsnemo.Module.from_checkpoint(
        "test/models/data/my_model_default_v1.0.mdlus"
    ).to(device)

    # Test attributes
    assert model.input_dim == 64
    assert model.output_dim == 32

    # Load reference and verify
    data = torch.load("test/models/data/my_model_default_v1.0.pth")
    x = data["x"].to(device)
    out_ref = data["out"].to(device)
    out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_my_model_public_method_non_regression(device):
    """Test public method compute_loss against reference."""
    model = _instantiate_model(MyModel, input_dim=64, output_dim=32).to(device)

    data = torch.load("test/models/data/my_model_loss_v1.0.pth")
    pred = data["pred"].to(device)
    target = data["target"].to(device)
    loss_ref = data["loss"].to(device)

    loss = model.compute_loss(pred, target)
    assert torch.allclose(loss, loss_ref, atol=1e-6, rtol=1e-6)
```

**Anti-pattern:**

```python
# WRONG: Only testing output shapes
def test_my_model_bad(device):
    model = MyModel(input_dim=64, output_dim=32).to(device)
    x = torch.randn(4, 64).to(device)
    out = model(x)
    assert out.shape == (4, 32)  # NOT SUFFICIENT!

# WRONG: Using singleton dimensions in test data
def test_my_model_bad(device):
    x = torch.randn(1, 1, 64)  # WRONG: Trivial shapes hide bugs

# WRONG: No parameterization
def test_my_model_bad():
    model = MyModel(input_dim=64, output_dim=32)  # Only tests defaults

# WRONG: No checkpoint loading test
# (Missing test_my_model_from_checkpoint entirely)

# WRONG: Public methods not tested
class MyModel(Module):
    def compute_loss(self, pred, target):  # No test for this method
        return F.mse_loss(pred, target)
```

---

## Compliance

When implementing models, ensure all rules are followed. Code reviews should
verify each rule is followed and enforce the rules as strictly as possible.
For exceptions to these rules, document the reasoning in code comments and
obtain approval during code review.
