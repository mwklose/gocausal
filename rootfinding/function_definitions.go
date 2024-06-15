package rootfinding

import (
	"gonum.org/v1/gonum/mat"
)

type VectorValuedFunction func(v mat.Vector) (mat.Vector, error)

type FunctionCall struct {
	mag float64
	vec mat.Vector
}
