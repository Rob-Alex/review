---
layout: post
title: "Testing Equation Numbering"
date: 2025-01-21
categories: testing math
---

# Testing KaTeX Equation Numbering

This post demonstrates the new equation numbering system that works seamlessly with your org-mode workflow.

## Basic Display Equations

Here's a simple display equation using `\[ \]` syntax (auto-numbered):

\[
E = mc^2
\]

And another one using equation environment:

\begin{equation}
F = ma
\end{equation}

## Labeled Equations

You can add labels in your org-mode files for cross-referencing:

\begin{equation}\label{maxwell}
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
\end{equation}

\begin{equation}\label{continuity}
\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{j} = 0
\end{equation}

## Cross-References

Now you can reference equations in your text! For example:

- Maxwell's equation \eqref{maxwell} describes electromagnetic induction
- The continuity equation \eqref{continuity} expresses charge conservation  
- Einstein's mass-energy relation was shown in equation (1)

## Inline Math (Unchanged)

Your inline math continues to work exactly as before: \( \alpha = \frac{\beta}{\gamma} \) and \( x^2 + y^2 = z^2 \).

## Complex Equations

The numbering works with complex multi-line equations too:

\begin{align}
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0\mathbf{j} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t} \\
\nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
\nabla \cdot \mathbf{B} &= 0
\end{align}

## Your Workflow Remains the Same

In your org-mode files, you just write:
```
\begin{equation}\label{newton}
F = ma
\end{equation}
```

Then reference with: `As shown in \eqref{newton}...`

The system automatically handles the numbering and creates clickable links!

The export process handles everything automatically!