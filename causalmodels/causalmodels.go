package causalmodels

import "gonum.org/v1/gonum/mat"

type Equation struct {
	lhs []string
	rhs []string
}

type CausalModel struct {
	variables  []string
	parameters []string
	f          map[string]Equation
}

type ModelEstimate struct {
	variables  []string
	parameters []float64
	f          map[string]Equation
	data       mat.Matrix
}

type ModelFitter interface {
	fit(data mat.Matrix) ModelEstimate
	display() string
}

type WeightedModelFitter interface {
	fit(data mat.Matrix, weights mat.Vector) ModelEstimate
}

type ModelPredictor interface {
	predict(newdata mat.Matrix, columns []string) mat.Matrix
}

type ModelIntervener interface {
	intervene(interventions map[string]Equation, columns []string) mat.Matrix
}
