package rootfinding

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func test_1d(x mat.Vector) (mat.Vector, error) {
	num := x.AtVec(0)
	result := mat.NewVecDense(1, []float64{num*num - 4})
	return result, nil
}

func test_2d_independent(x mat.Vector) (mat.Vector, error) {
	x0, x1 := x.AtVec(0), x.AtVec(1)
	result := mat.NewVecDense(2, []float64{
		x0*x0 - x0 - 6,
		math.Pow(x1, 3) - 2*math.Pow(x1, 2) - 19*x1 + 20,
	})
	return result, nil
}

func test_2d_dependent(x mat.Vector) (mat.Vector, error) {
	x0, x1 := x.AtVec(0), x.AtVec(1)
	result := mat.NewVecDense(2, []float64{
		x0*x0*x1 - 12,
		x1 - x0 - 1,
	})
	return result, nil
}

func TestSimplexRoots1D(t *testing.T) {
	rootl, rootr := -2.0, 2.0

	init_cond := mat.NewVecDense(1, []float64{1})
	bounds1 := mat.NewDense(1, 2, []float64{-1.5, 10})

	final, err := SimplexRoots(test_1d, init_cond, bounds1)
	if err != nil {
		t.Errorf("1D case (upper root) returned error %s", err)
	}

	if math.Pow(final.AtVec(0)-rootr, 2) > 1e-9 {
		t.Errorf("Square difference greater than 1e-9; received %f", final.AtVec(0))
	}

	bounds2 := mat.NewDense(1, 2, []float64{-5, 1.5})

	final, err = SimplexRoots(test_1d, init_cond, bounds2)
	if err != nil {
		t.Errorf("1D case (lower root) returned error %s", err)
	}

	if math.Pow(final.AtVec(0)-rootl, 2) > 1e-9 {
		t.Errorf("Square difference greater than 1e-9; received %f", final.AtVec(0))
	}

}

func TestSimplexRoots2D(t *testing.T) {
	init_cond := mat.NewVecDense(2, []float64{1, 1})
	bounds1 := mat.NewDense(2, 2, []float64{-1, 10, -3.5, 3.5})

	roots_indep := []float64{3.0, 1.0}

	final, err := SimplexRoots(test_2d_independent, init_cond, bounds1)
	if err != nil {
		t.Errorf("2D independent case returned error %s", err)
	}

	result := 0.0
	length := init_cond.Len()
	for i := range length {
		diff := final.AtVec(i) - roots_indep[i]
		result += diff * diff
	}
	if result > 1e-9 {
		t.Errorf("Square difference of dependent greater than 1e-9; received %f", result)
	}

	roots_dep := []float64{2.0, 3.0}
	init_cond = mat.NewVecDense(2, []float64{3.0, 4.0})
	bounds2 := mat.NewDense(2, 2, []float64{0.0, 12.0, 0.0, 12.0})

	final, err = SimplexRoots(test_2d_dependent, init_cond, bounds2)
	if err != nil {
		t.Errorf("2D dependent case returned error %s", err)
	}

	result = 0.0
	for i := range length {
		diff := final.AtVec(i) - roots_dep[i]
		result += diff * diff
	}
	if result > 1e-9 {
		t.Errorf("Square difference of independent greater than 1e-9; received %f", result)
	}

}
