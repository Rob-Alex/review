---
layout: post
title: "Testing Modern Syntax Highlighting"
date: 2025-01-21
categories: testing
---

# Modern Code Block Testing

Testing the new Prism.js syntax highlighting with your preferred languages:

## C++ Example

```cpp
#include <iostream>
#include <vector>
#include <memory>

template<typename T>
class TensorBuffer {
private:
    std::unique_ptr<T[]> data_;
    size_t size_;

public:
    TensorBuffer(size_t size) : size_(size), data_(std::make_unique<T[]>(size)) {}
    
    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[index];
    }
    
    void fill(const T& value) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    }
};

int main() {
    TensorBuffer<float> buffer(1000);
    buffer.fill(0.0f);
    
    std::cout << "Buffer created with size: " << buffer.size() << std::endl;
    return 0;
}
```

## Python Example

```python
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt

class FiniteVolumeMethod:
    def __init__(self, grid_size: int, boundary_conditions: dict):
        self.grid_size = grid_size
        self.boundary_conditions = boundary_conditions
        self.solution = np.zeros(grid_size)
    
    def solve_poisson(self, source_term: np.ndarray, 
                     tolerance: float = 1e-6) -> np.ndarray:
        """
        Solve Poisson equation using finite volume method
        ∇²φ = f
        """
        max_iterations = 10000
        
        for iteration in range(max_iterations):
            old_solution = self.solution.copy()
            
            # Interior points
            for i in range(1, self.grid_size - 1):
                self.solution[i] = 0.5 * (
                    self.solution[i-1] + self.solution[i+1] - source_term[i]
                )
            
            # Apply boundary conditions
            self._apply_boundaries()
            
            # Check convergence
            residual = np.max(np.abs(self.solution - old_solution))
            if residual < tolerance:
                print(f"Converged in {iteration} iterations")
                break
        
        return self.solution
    
    def _apply_boundaries(self):
        """Apply Dirichlet boundary conditions"""
        self.solution[0] = self.boundary_conditions.get('left', 0.0)
        self.solution[-1] = self.boundary_conditions.get('right', 0.0)
```

## Rust Example

```rust
use std::fmt::Display;

trait ComputeBackend {
    type Buffer;
    type Error;
    
    fn allocate_buffer(&self, size: usize) -> Result<Self::Buffer, Self::Error>;
    fn copy_to_device(&self, data: &[f32], buffer: &mut Self::Buffer) -> Result<(), Self::Error>;
    fn execute_kernel(&self, kernel_name: &str, buffer: &Self::Buffer) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct MetalBackend {
    device: metal::Device,
    command_queue: metal::CommandQueue,
}

impl MetalBackend {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device = metal::Device::system_default()
            .ok_or("No Metal device found")?;
        
        let command_queue = device.new_command_queue();
        
        Ok(MetalBackend { device, command_queue })
    }
    
    pub fn create_compute_pipeline(&self, source: &str) -> Result<metal::ComputePipelineState, Box<dyn std::error::Error>> {
        let library = self.device.new_library_with_source(source, &metal::CompileOptions::new())?;
        let function = library.get_function("saxpy", None)?;
        
        Ok(self.device.new_compute_pipeline_state_with_function(&function)?)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = MetalBackend::new()?;
    println!("Metal backend initialized successfully");
    Ok(())
}
```

## Common Lisp Example

```lisp
(defpackage :tensor-lib
  (:use :cl)
  (:export #:tensor #:make-tensor #:tensor-ref #:tensor-dimensions
           #:matrix-multiply #:element-wise-add))

(in-package :tensor-lib)

(defclass tensor ()
  ((data :initarg :data :accessor tensor-data)
   (dimensions :initarg :dimensions :accessor tensor-dimensions)
   (backend :initarg :backend :accessor tensor-backend :initform :cpu)))

(defmethod make-tensor (dimensions &key (initial-value 0.0) (backend :cpu))
  "Create a new tensor with given dimensions"
  (let ((total-size (reduce #'* dimensions)))
    (make-instance 'tensor
                   :data (make-array total-size :initial-element initial-value
                                   :element-type 'single-float)
                   :dimensions dimensions
                   :backend backend)))

(defmethod tensor-ref ((tensor tensor) &rest indices)
  "Reference a tensor element by indices"
  (let ((flat-index (compute-flat-index indices (tensor-dimensions tensor))))
    (aref (tensor-data tensor) flat-index)))

(defun compute-flat-index (indices dimensions)
  "Convert multi-dimensional indices to flat array index"
  (let ((index 0)
        (multiplier 1))
    (loop for i from (1- (length indices)) downto 0
          for dim-i from (1- (length dimensions)) downto 0
          do (incf index (* (nth i indices) multiplier))
             (setf multiplier (* multiplier (nth dim-i dimensions))))
    index))

(defmethod matrix-multiply ((a tensor) (b tensor))
  "Multiply two 2D tensors (matrices)"
  (let ((a-dims (tensor-dimensions a))
        (b-dims (tensor-dimensions b)))
    (assert (= (second a-dims) (first b-dims)) ()
            "Matrix dimensions incompatible for multiplication")
    
    (let ((result (make-tensor (list (first a-dims) (second b-dims)))))
      (loop for i from 0 below (first a-dims)
            do (loop for j from 0 below (second b-dims)
                     do (loop for k from 0 below (second a-dims)
                              do (incf (tensor-ref result i j)
                                     (* (tensor-ref a i k)
                                        (tensor-ref b k j))))))
      result)))

;; Example usage
(let ((matrix-a (make-tensor '(3 4)))
      (matrix-b (make-tensor '(4 2))))
  (format t "Created matrices: ~A x ~A~%" 
          (tensor-dimensions matrix-a)
          (tensor-dimensions matrix-b))
  (matrix-multiply matrix-a matrix-b))
```

## Lua Example

```lua
-- Tensor library for scientific computing
local Tensor = {}
Tensor.__index = Tensor

function Tensor:new(dimensions, backend)
    local obj = {
        dimensions = dimensions or {},
        backend = backend or "cpu",
        data = {}
    }
    
    -- Calculate total size
    local total_size = 1
    for _, dim in ipairs(dimensions) do
        total_size = total_size * dim
    end
    
    -- Initialize data
    for i = 1, total_size do
        obj.data[i] = 0.0
    end
    
    setmetatable(obj, self)
    return obj
end

function Tensor:size()
    return #self.data
end

function Tensor:reshape(new_dimensions)
    local total_size = 1
    for _, dim in ipairs(new_dimensions) do
        total_size = total_size * dim
    end
    
    assert(total_size == #self.data, "Cannot reshape: size mismatch")
    self.dimensions = new_dimensions
    return self
end

function Tensor:get(...)
    local indices = {...}
    local flat_index = self:_compute_flat_index(indices)
    return self.data[flat_index]
end

function Tensor:set(value, ...)
    local indices = {...}
    local flat_index = self:_compute_flat_index(indices)
    self.data[flat_index] = value
end

function Tensor:_compute_flat_index(indices)
    local index = 0
    local multiplier = 1
    
    for i = #indices, 1, -1 do
        index = index + (indices[i] - 1) * multiplier
        multiplier = multiplier * self.dimensions[i]
    end
    
    return index + 1  -- Lua uses 1-based indexing
end

function Tensor:apply(func)
    for i = 1, #self.data do
        self.data[i] = func(self.data[i])
    end
    return self
end

function Tensor:__tostring()
    return string.format("Tensor%s [%s backend]", 
                        table.concat(self.dimensions, "x"), 
                        self.backend)
end

-- Example usage
local function main()
    local tensor = Tensor:new({3, 4, 2}, "metal")
    print("Created:", tensor)
    
    -- Fill with some values
    local value = 1.0
    for i = 1, tensor:size() do
        tensor.data[i] = value
        value = value + 0.1
    end
    
    -- Apply a function
    tensor:apply(function(x) return math.sin(x) end)
    
    print("Applied sine function to all elements")
    print("Element at (1,1,1):", tensor:get(1, 1, 1))
end

main()
```

This demonstrates the new modern syntax highlighting with copy buttons, line numbers, and clean styling for all your preferred languages!