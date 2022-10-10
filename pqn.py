'''
Source: https://github.com/mlds-lab/ems/tree/master/MLE/pyPQN
'''
from __future__ import division
import numpy as np
from numpy.linalg import norm, solve


def minConF_SPG(funObj, x, funProj,
                verbose=2,
                numDiff=0,
                optTol=1e-6,
                maxIter=500,
                suffDec=1e-4,
                interp=2,
                memory=10,
                useSpectral=1,
                curvilinear=0,
                feasibleInit=0,
                testOpt=1,
                bbType=1):

    """Function for using Spectral Projected Gradient to solve problems
    of the form:
    min funObj(x) s.t. x in C
    @funObj(x):
        function to minimize (returns gradient as second argument)
    @funProj(x):
        function that returns projection of x onto C
    options:
        verbose:
            level of verbosity
            (0: no output, 1: final, 2: iter (default), 3: debug)
        optTol:
            tolerance used to check for progress (default: 1e-6)
        maxIter:
            maximum number of calls to funObj (default: 500)
        numDiff:
            compute derivatives numerically
            (0: use user-supplied derivatives (default),
             1: use finite differences,
             2: use complex differentials)
        suffDec:
            sufficient decrease parameter in Armijo condition (default: 1e-4)
        interp:
            type of interpolation
            (0: step-size halving, 1: quadratic, 2: cubic)
        memory:
            number of steps to look back in non-monotone Armijo condition
        useSpectral:
            use spectral scaling of gradient direction (default: 1)
        curvilinear:
            backtrack along projection Arc (default: 0)
        testOpt:
            test optimality condition (default: 1)
        feasibleInit:
            if 1, then the initial point is assumed to be feasible
        bbType:
            type of Barzilai Borwein step (default: 1)
    Notes:
        - if the projection is expensive to compute, you can reduce the
          number of projections by setting testOpt to 0
    """

    #nVars = len(x)

    # Output Log
    if verbose >= 2:
        if testOpt:
            print('%10s %10s %10s %15s %15s %15s' % (
                    'Iteration', 'FunEvals', 'Projections',
                    'Step Length', 'Function Val', 'Opt Cond'))
        else:
            print('%10s %10s %10s %15s %15s' % (
                    'Iteration', 'FunEvals', 'Projections',
                    'Step Length', 'Function Val'))

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1
    # FIXME
    #if numDiff:
    #    if numDiff == 2:
    #        useComplex = 1
    #    else:
    #        useComplex = 0
    #    funObj = @(x)autoGrad(x,useComplex,funObj)
    #    funEvalMultiplier = nVars+1-useComplex
    #end

    # Evaluate Initial Point
    if not feasibleInit:
        x = funProj(x)

    f, g = funObj(x)
    projects = 1
    funEvals = 1

    # Optionally check optimality
    if testOpt:
        projects += 1
        if norm(funProj(x - g) - x, 1) < optTol:
            if verbose >= 1:
                print('First-Order Optimality Conditions Below optTol at Initial Point')
            return x, f, funEvals, projects

    i = 0
    f_prev, t_prev = 0, 0

    while funEvals <= maxIter:

        # Compute Step Direction
        if i == 0 or not useSpectral:
            alpha = 1
        else:
            y = g - g_old
            s = x - x_old
            if bbType == 1:
                alpha = s.dot(s) / s.dot(y)
            else:
                alpha = s.dot(y) / y.dot(y)

            if alpha <= 1e-10 or alpha > 1e10:
                alpha = 1

        d = -alpha * g
        f_old = f
        x_old = x
        g_old = g

        # Compute Projected Step
        if not curvilinear:
            d = funProj(x + d) - x
            projects += 1

        # Check that Progress can be made along the direction
        gtd = g.dot(d)
        if gtd > -optTol:
            if verbose >= 1:
                print('Directional Derivative below optTol')
            break

        # Select Initial Guess to step length
        if i == 0:
            t = min(1, 1 / norm(g, 1))
        else:
            t = 1

        # Compute reference function for non-monotone condition

        if memory == 1:
            funRef = f
        else:
            if i == 0:
                old_fvals = np.empty(memory)
                old_fvals.fill(-np.inf)

            if i < memory:
                old_fvals[i] = f
            else:
                old_fvals = np.concatenate((old_fvals[1:], np.array([f])))

            funRef = old_fvals.max()

        # Evaluate the Objective and Gradient at the Initial Step
        if curvilinear:
            x_new = funProj(x + t * d)
            projects += 1
        else:
            x_new = x + t * d

        f_new, g_new = funObj(x_new)
        funEvals += 1

        # Backtracking Line Search
        lineSearchIters = 1
        while f_new > funRef + suffDec * g.dot(x_new-x) or not isLegal(f_new):
            temp = t
            if interp == 0 or not isLegal(f_new):
                if verbose == 3:
                    print('Halving Step Size')
                t = t / 2
            elif interp == 2 and isLegal(g_new):
                if verbose == 3:
                    print('Cubic Backtracking')
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, g_new.dot(d)]]))
            elif lineSearchIters < 2 or not isLegal(f_prev):
                if verbose == 3:
                    print('Quadratic Backtracking')
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, np.sqrt(-1)]]))
            else:
                if verbose == 3:
                    print('Cubic Backtracking on Function Values')
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, np.sqrt(-1)],
                                         [t_prev, f_prev, np.sqrt(-1)]]))

            # Adjust if change is too small
            if t < temp * 1e-3:
                if verbose == 3:
                    print('Interpolated value too small, Adjusting')
                t = temp * 1e-3
            elif t > temp * 0.6:
                if verbose == 3:
                    print('Interpolated value too large, Adjusting')
                t = temp * 0.6

            # Check whether step has become too small
            if norm(t * d, 1) < optTol or t == 0:
                if verbose == 3:
                    print('Line Search failed')
                t = 0
                f_new = f
                g_new = g
                break

            # Evaluate New Point
            f_prev = f_new
            t_prev = temp
            if curvilinear:
                x_new = funProj(x + t * d)
                projects += 1
            else:
                x_new = x + t * d
            f_new, g_new = funObj(x_new)
            funEvals += 1
            lineSearchIters += 1

        # Take Step
        x = x_new
        f = f_new
        g = g_new

        if testOpt:
            optCond = norm(funProj(x - g) - x, 1)
            projects += 1

        # Output Log
        if verbose >= 2:
            if testOpt:
                print('%10d %10d %10d %15.5e %15.5e %15.5e' % (
                        i, funEvals * funEvalMultiplier, projects,
                        t, f, optCond))
            else:
                print('%10d %10d %10d %15.5e %15.5e' % (
                        i, funEvals * funEvalMultiplier, projects, t, f))

        # Check optimality
        if testOpt:
            if optCond < optTol:
                if verbose >= 1:
                    print('First-Order Optimality Conditions Below optTol')
                break

        if norm(t * d, 1) < optTol:
            if verbose >= 1:
                print('Step size below optTol')
            break

        if np.fabs(f - f_old) < optTol:
            if verbose >= 1:
                print('Function value changing by less than optTol')
            break

        if funEvals * funEvalMultiplier > maxIter:
            if verbose >= 1:
                print('Function Evaluations exceeds maxIter')
            break

        i += 1

    return x, f, funEvals, projects


def polyinterp(points):
    """Minimum of interpolating polynomial based on function and derivative
    values
    In can also be used for extrapolation if {xmin,xmax} are outside
    the domain of the points.
    Input:
        points(pointNum,[x f g])
        xmin: min value that brackets minimum (default: min of points)
        xmax: max value that brackets maximum (default: max of points)
    set f or g to sqrt(-1) if they are not known
    the order of the polynomial is the number of known f and g values minus 1
    """

    nPoints = points.shape[0]
    order = (np.isreal(points[:, 1:3])).sum() - 1

    # Code for most common case:
    #   - cubic interpolation of 2 points
    #       w/ function and derivative values for both
    #   - no xminBound/xmaxBound

    if nPoints == 2 and order == 3:
        # Solution in this case (where x2 is the farthest point):
        #    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2)
        #    d2 = sqrt(d1^2 - g1*g2)
        #    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        #    t_new = min(max(minPos,x1),x2)
        if points[0, 1] < points[1, 1]:
            x_lo, x_hi = points[0, 0], points[1, 0]
            f_lo, f_hi = points[0, 1], points[1, 1]
            g_lo, g_hi = points[0, 2], points[1, 2]
        else:
            x_lo, x_hi = points[1, 0], points[0, 0]
            f_lo, f_hi = points[1, 1], points[0, 1]
            g_lo, g_hi = points[1, 2], points[0, 2]
        d1 = g_lo + g_hi - 3 * (f_lo - f_hi) / (x_lo - x_hi)
        d2 = np.sqrt(d1 * d1 - g_lo * g_hi)
        if np.isreal(d2):
            t = x_hi - (x_hi - x_lo) * ((g_hi + d2 - d1) /
                                        (g_hi - g_lo + 2 * d2))
            minPos = min(max(t, x_lo), x_hi)
        else:
            minPos = (x_lo + x_hi) / 2
        return minPos

    xmin = min(points[:, 0])
    xmax = max(points[:, 0])

    # Compute Bounds of Interpolation Area

    xminBound = xmin
    xmaxBound = xmax

    # Constraints Based on available Function Values
    A = np.zeros((0, order + 1))
    b = []

    for i in xrange(nPoints):
        if np.isreal(points[i, 1]):
            constraint = np.zeros(order + 1)
            for j in xrange(order + 1):
                constraint[order - j] = points[i, 0]**j
            A = np.vstack((A, constraint))
            b = np.append(b, points[i, 1])

    # Constraints based on available Derivatives
    for i in xrange(nPoints):
        if np.isreal(points[i, 2]):
            constraint = np.zeros(order + 1)
            for j in xrange(order):
                constraint[j] = (order - j) * points[i, 0]**(order - j - 1)
            A = np.vstack((A, constraint))
            b = np.append(b, points[i, 2])

    # Find interpolating polynomial
    params = solve(A, b)

    # Compute Critical Points
    dParams = np.zeros(order)
    for i in xrange(len(params) - 1):
        dParams[i] = params[i] * (order - i)

    if np.any(np.isinf(dParams)):
        cp = np.concatenate((np.array([xminBound, xmaxBound]),
                             points[:, 0]))
    else:
        cp = np.concatenate((np.array([xminBound, xmaxBound]),
                             points[:, 0]),
                             np.roots(dParams))

    # Test Critical Points
    fmin = np.inf
    # Default to Bisection if no critical points valid
    minPos = (xminBound + xmaxBound) / 2
    for xCP in cp:
        if np.isreal(xCP) and xCP >= xminBound and xCP <= xmaxBound:
            fCP = np.polyval(params, xCP)
            if np.isreal(fCP) and fCP < fmin:
                minPos = np.real(xCP)
                fmin = np.real(fCP)

    return minPos


def isLegal(v):
    return (np.all(np.isreal(v)) and
            not np.any(np.isnan(v)) and
            not np.any(np.isinf(v)))


def lbfgsUpdate(y, s, corrections, debug, old_dirs, old_stps, Hdiag):
    if y.dot(s) > 1e-10:
        numVars, numCorrections = old_dirs.shape
        if numCorrections < corrections:
            # Full Update
            new_dirs = np.empty((numVars, numCorrections + 1))
            new_stps = np.empty((numVars, numCorrections + 1))
            new_dirs[:, :-1] = old_dirs
            new_stps[:, :-1] = old_stps
        else:
            # Limited-Memory Update
            new_dirs = np.empty((numVars, corrections))
            new_stps = np.empty((numVars, corrections))
            new_dirs[:, :-1] = old_dirs[:, 1:]
            new_stps[:, :-1] = old_stps[:, 1:]
        new_dirs[:, -1] = s
        new_stps[:, -1] = y

        # Update scale of initial Hessian approximation
        Hdiag = y.dot(s) / y.dot(y)
    else:
        if debug:
            print('Skipping Update')
        new_dirs = old_dirs
        new_stps = old_stps

    return new_dirs, new_stps, Hdiag


def minConf_PQN(funObj, x, funProj,
                verbose=2,
                optTol=1e-6,
                maxIter=500,
                maxProject=100000,
                numDiff=0,
                suffDec=1e-4,
                corrections=10,
                adjustStep=0,
                bbInit=1,
                SPGoptTol=1e-6,
                SPGiters=10,
                SPGtestOpt=0):
    """Function for using a limited-memory projected quasi-Newton to solve
    problems of the form:

    min funObj(x) s.t. x in C

    The projected quasi-Newton sub-problems are solved the spectral projected
    gradient algorithm

    @funObj(x):
        function to minimize (returns gradient as second argument)
    @funProj(x):
        function that returns projection of x onto C

    options:

    verbose:
        level of verbosity
        (0: no output, 1: final, 2: iter (default), 3: debug)
    optTol:
        tolerance used to check for progress (default: 1e-6)
    maxIter:
        maximum number of calls to funObj (default: 500)
    maxProject:
        maximum number of calls to funProj (default: 100000)
    numDiff:
        compute derivatives numerically
        (0: use user-supplied derivatives (default),
         1: use finite differences,
         2: use complex differentials)
    suffDec:
        sufficient decrease parameter in Armijo condition (default: 1e-4)
    corrections:
        number of lbfgs corrections to store (default: 10)
    adjustStep:
        use quadratic initialization of line search (default: 0)
    bbInit:
        initialize sub-problem with Barzilai-Borwein step (default: 1)
    SPGoptTol:
        optimality tolerance for SPG direction finding (default: 1e-6)
    SPGiters:
        maximum number of iterations for SPG direction finding (default: 10)
    """

    nVars = len(x)

    # Output Parameter Settings
    if verbose >= 3:
        print('Running PQN...')
        print('Number of L-BFGS Corrections to store: %d' % corrections)
        print('Spectral initialization of SPG: %d' % bbInit)
        print('Maximum number of SPG iterations: %d' % SPGiters)
        print('SPG optimality tolerance: %.2e' % SPGoptTol)
        print('PQN optimality tolerance: %.2e' % optTol)
        print('Quadratic initialization of line search: %d' % adjustStep)
        print('Maximum number of function evaluations: %d' % maxIter)
        print('Maximum number of projections: %d' % maxProject)

    # Output Log
    if verbose >= 2:
        print('%10s %10s %10s %15s %15s %15s' % ('Iteration', 'FunEvals', 'Projection', 'Step Length', 'Function Val', 'Opt Cond'))

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1
    # FIXME implement autoGrad
    #if numDiff:
    #    if numDiff == 2:
    #        useComplex = 1
    #    else:
    #        useComplex = 0
    #    funObj = @(x)autoGrad(x,useComplex,funObj)
    #    funEvalMultiplier = nVars+1-useComplex

    # Project initial parameter vector
    # print('='*20, 'minConf_PQN: calling funProj', x)
    x = funProj(x)
    projects = 1
    # print('='*20, 'minConf_PQN: calling funObj', x)
    # Evaluate initial parameters
    f, g = funObj(x)
    funEvals = 1
    # print('='*20, 'minConf_PQN: finished funObj', f,g)
    
    # Check Optimality of Initial Point
    projects += 1

    # print('=' * 20, 'minConf_PQN: first return attempt')

    if norm(funProj(x - g) - x, 1) < optTol:
        if verbose >= 1:
            print('First-Order Optimality Conditions Below optTol at Initial Point')
        return x, f, funEvals

    f_old, g_old, x_old = 0, 0, 0
    i = 1

    # print('=' * 20, 'minConf_PQN: starting while loop')

    while funEvals <= maxIter:

        # Compute Step Direction
        if i == 1:
            p = funProj(x - g)
            projects += 1
            S = np.zeros((nVars, 0))
            Y = np.zeros((nVars, 0))
            Hdiag = 1
        else:
            y = g - g_old
            s = x - x_old

            # print('=' * 20, 'minConf_PQN: calling lbfgsUpdate:')

            S, Y, Hdiag = lbfgsUpdate(y, s, corrections, verbose==3,
                                      S, Y, Hdiag)

            # print('=' * 20, 'minConf_PQN: exiting lbfgsUpdate.')

            # Make Compact Representation
            k = Y.shape[1]
            L = np.zeros((k, k))
            for j in range(k):
                L[(j + 1):, j] = S[:, (j + 1):].T.dot(Y[:, j])

            # print('=' * 20, 'minConf_PQN: Right before hstack')

            N = np.hstack((S / Hdiag, Y))
            #M = np.vstack((np.hstack((S.T.dot(S) / Hdiag, L)),
            #               np.hstack((L.T, -np.diag(np.diag(S.T.dot(Y)))))))
            M = np.empty((k * 2, k * 2))
            M[:k, :k] = S.T.dot(S) / Hdiag
            M[:k, k:] = L
            M[k:, :k] = L.T
            M[k:, k:] = -np.diag(np.diag(S.T.dot(Y)))

            def HvFunc(v):
                return v / Hdiag - N.dot(solve(M, (N.T.dot(v))))

            if bbInit:
                # Use Barzilai-Borwein step to initialize sub-problem
                alpha = s.dot(s) / s.dot(y)
                if alpha <= 1e-10 or alpha > 1e10:
                    alpha = 1 / norm(g)

                # Solve Sub-problem
                # FIXME
                #xSubInit = x - alpha * g
                feasibleInit = 0
            else:
                # FIXME
                #xSubInit = x
                feasibleInit = 1

            # print('=' * 20, 'minConf_PQN: Right before solveSubProblem')
            # print(f'x={x}, g={g}')

            # Solve Sub-problem
            p, subProjects = solveSubProblem(x, g, HvFunc, funProj, SPGoptTol,
                                             SPGiters, SPGtestOpt,
                                             feasibleInit, x)
            projects += subProjects

            # print('=' * 20, 'minConf_PQN: Right after solveSubProblem')

        d = p - x
        g_old = g
        x_old = x

        # Check that Progress can be made along the direction
        gtd = g.dot(d)
        if gtd > -optTol:
            if verbose >= 1:
                print('Directional Derivative below optTol')
            break

        # print('=' * 20, 'minConf_PQN: Select Initial Guess')
        # print(f'gtd={gtd}')

        # Select Initial Guess to step length
        if i == 1 or adjustStep == 0:
            t = 1
        else:
            t = min(1, 2 * (f - f_old) / gtd)

        # print(f'norm_of_g={norm(g, 1)}')

        # Bound Step length on first iteration
        if i == 1:
            t = min(1, 1 / norm(g, 1))

        # print('=' * 20, 'minConf_PQN: Evaluate the Objective and Gradient at the Initial Step')

        # Evaluate the Objective and Gradient at the Initial Step
        x_new = x + t * d

        # print(f'x_new={x_new}')

        f_new, g_new = funObj(x_new)
        funEvals += 1

        # print('=' * 20, 'minConf_PQN: Backtracking Line Search')

        # Backtracking Line Search
        f_old = f
        while f_new > f + suffDec * t * gtd or not isLegal(f_new):
            temp = t

            # Backtrack to next trial value
            if not (isLegal(f_new) and isLegal(g_new)):
                if verbose == 3:
                    print('Halving Step Size')
                t /= 2
            else:
                if verbose == 3:
                    print('Cubic Backtracking')
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, g_new.dot(d)]]))

            # Adjust if change is too small/large
            if t < temp * 1e-3:
                if verbose == 3:
                    print('Interpolated value too small, Adjusting')
                t = temp * 1e-3
            elif t > temp * 0.6:
                if verbose == 3:
                    print('Interpolated value too large, Adjusting')
                t = temp * 0.6

            # Check whether step has become too small
            if i > 1 and (norm(t * d, 1) < optTol or t == 0):
                if verbose == 3:
                    print('Line Search failed')
                t = 0
                f_new = f
                g_new = g
                break

            # print('=' * 20, 'minConf_PQN: Evaluate New Point')

            # Evaluate New Point
            x_new = x + t * d
            f_new, g_new = funObj(x_new)
            funEvals += 1
        
        # Take Step
        x = x_new
        f = f_new
        g = g_new

        optCond = sum(abs(funProj(x-g)-x))
        optCond = norm(funProj(x - g) - x, 1)
        projects += 1

        # Output Log
        if verbose >= 2:
            print('%10d %10d %10d %15.5e %15.5e %15.5e' % (
                    i, funEvals * funEvalMultiplier, projects, t, f, optCond))

        # Check optimality
        if optCond < optTol:
            print('First-Order Optimality Conditions Below optTol')
            break

        if norm(t * d, 1) < optTol:
            if verbose >= 1:
                print('Step size below optTol')
            break

        if np.fabs(f - f_old) < optTol:
            if verbose >= 1:
                print('Function value changing by less than optTol')
            break

        if funEvals * funEvalMultiplier > maxIter:
            if verbose >= 1:
                print('Function Evaluations exceeds maxIter')
            break

        if projects > maxProject:
            if verbose >= 1:
                print('Number of projections exceeds maxProject')
            break

        i += 1

    return x, f, funEvals


def solveSubProblem(x, g, HvFunc, funProj, optTol, maxIter, testOpt,
                    feasibleInit, x_init):
    # Uses SPG to solve for projected quasi-Newton direction

    def subHv(p):
        d = p - x
        Hd = HvFunc(d)
        return g.dot(d) + d.dot(Hd) / 2, g + Hd

    p, f, funEvals, subProjects = minConF_SPG(subHv, x_init, funProj,
                                              verbose=0,
                                              optTol=optTol,
                                              maxIter=maxIter,
                                              testOpt=testOpt,
                                              feasibleInit=feasibleInit)
    return p, subProjects

