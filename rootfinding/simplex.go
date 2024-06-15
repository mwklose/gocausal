package rootfinding

import (
	"errors"
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

const (
	MAX_ITERS = 10000
	TOL       = 1e-16
)

func SquaredMagnitude(arr mat.Vector) float64 {
	mag := 0.0
	vec_len := arr.Len()
	for i := range vec_len {
		num := arr.AtVec(i)
		square := num * num
		mag += square
	}
	return mag
}

func sign_int(arr mat.Vector) int {
	result := 0
	vec_len := arr.Len()
	for i := range vec_len {
		if arr.AtVec(i) > 0 {
			result |= (1 << i)
		}
	}

	return result
}

func opposite_sign_int(arr mat.Vector) int {
	result := 0
	vec_len := arr.Len()
	for i := range vec_len {
		if arr.AtVec(i) <= 0 {
			result |= (1 << i)
		}
	}

	return result
}

func insert_into_lookup_table(lt map[int]FunctionCall, res mat.Vector, sign int, magnitude float64) (lookup map[int]FunctionCall) {
	elem, ok := lt[sign]
	if !ok || (elem.mag >= magnitude) {
		lt[sign] = FunctionCall{magnitude, res}
	}

	return lt
}

func iterative_simplex_roots(f VectorValuedFunction, x mat.Vector, lookup_table map[int]FunctionCall) (magnitude float64, next_cond mat.Vector, next_lookup_table map[int]FunctionCall, err error) {
	eval, err := f(x)

	if err != nil {
		return -1, x, lookup_table, err
	}

	sign_eval := sign_int(eval)
	squared_magnitude_eval := SquaredMagnitude(eval)

	lookup_table = insert_into_lookup_table(lookup_table, x, sign_eval, squared_magnitude_eval)
	// If less than tolerance, finish
	if squared_magnitude_eval < TOL {
		return squared_magnitude_eval, x, lookup_table, nil
	}
	// Otherwise, need to try other values, so start by:
	// Find the most opposite value
	opposite, ok := lookup_table[opposite_sign_int(eval)]
	if !ok {
		min_idx := -1
		min_value := opposite_sign_int(eval)

		for k := range lookup_table {
			if (sign_eval ^ k) <= min_value {
				min_idx = k
				min_value = (sign_eval ^ k)
			}
		}

		// TODO: handle random lookup if not present?
		opposite, ok = lookup_table[min_idx]
		if !ok {
			return -1, nil, nil, fmt.Errorf("[signeval=%d min_idx=%d min_value=%d]lookup table should be pre-populated with at least one value", sign_eval, min_idx, min_value)
		}
	}

	next_cond_vec := mat.VecDenseCopyOf(x)
	next_cond_vec.AddVec(x, opposite.vec)
	divide := make([]float64, next_cond_vec.Len())
	for i := range divide {
		divide[i] = 2
	}
	next_cond_vec.DivElemVec(next_cond_vec, mat.NewVecDense(next_cond_vec.Len(), divide))

	// Return everything
	return squared_magnitude_eval, next_cond_vec, lookup_table, nil
}

func SimplexRoots(f VectorValuedFunction, initial_conditions mat.Vector, bounds mat.Matrix) (mat.Vector, error) {
	// Argument verification
	length := initial_conditions.Len()
	rows, cols := bounds.Dims()
	if rows != length {
		return nil, errors.New("number of rows not equal to initial condition length")
	}
	if cols != 2 {
		return nil, errors.New("number of columns not equal to 2, corresponding to the bounds")
	}

	output, err := f(initial_conditions)
	if err != nil {
		return nil, err
	}

	// First, make the lookup table:
	lookup_table := make(map[int]FunctionCall)

	// Then, evaluate the vector valued function at a random sample of the boundary values
	for i := 0; i < length; i++ {
		lower_eval, upper_eval := make([]float64, length), make([]float64, length)
		lower_eval = mat.Col(lower_eval, 0, bounds)
		upper_eval = mat.Col(upper_eval, 1, bounds)

		for j := 0; j < length; j++ {
			if i == j {
				continue
			}
			random_lower, random_upper := rand.Intn(2), rand.Intn(2)
			lower_eval[j] = bounds.At(j, random_lower)
			upper_eval[j] = bounds.At(j, random_upper)
		}

		lower_eval_vec, upper_eval_vec := mat.NewVecDense(length, lower_eval), mat.NewVecDense(length, upper_eval)
		lower_res, err := f(lower_eval_vec)
		if err != nil {
			return nil, err
		}
		lower_sign := sign_int(lower_res)
		lower_magnitude := SquaredMagnitude(lower_res)
		lookup_table = insert_into_lookup_table(lookup_table, lower_eval_vec, lower_sign, lower_magnitude)

		upper_res, err := f(upper_eval_vec)
		if err != nil {
			return nil, err
		}
		upper_sign := sign_int(upper_res)
		upper_magnitude := SquaredMagnitude(upper_res)
		lookup_table = insert_into_lookup_table(lookup_table, upper_eval_vec, upper_sign, upper_magnitude)
	}

	// Lastly, keep iterating until max iterations are reached
	iters := 0
	squared_magnitude_output := SquaredMagnitude(output)

	next_conditions := mat.VecDenseCopyOf(initial_conditions).TVec()

	for (iters < MAX_ITERS) && (squared_magnitude_output > TOL) {

		squared_magnitude_output, next_conditions, lookup_table, err = iterative_simplex_roots(f, next_conditions, lookup_table)
		if err != nil {
			return nil, err
		}
		iters += 1
	}

	if squared_magnitude_output > TOL {
		return nil, errors.New("minimum tolerance not reached within number of iterations")
	}

	return next_conditions, nil
}
