
import numpy as np


def solve_policy_lp(losses:list[float],costs:list[float],budgets:list[float],rtol:float=1e-6,atol:float=1e-5):
    import cvxpy as cp

    losses_np=np.array(losses,dtype='float64')
    costs_np=np.array(costs,dtype='float64')
    cost_scale=max(float(np.max(np.abs(costs_np))),1.0)
    scaled_costs=costs_np/cost_scale
    rows=[]
    for budget in budgets:
        scaled_budget=budget/cost_scale
        x=cp.Variable(len(losses_np),nonneg=True)
        simplex=sum(x)==1
        budget_constraint=scaled_costs@x<=scaled_budget
        problem=cp.Problem(cp.Minimize(losses_np@x),[simplex,budget_constraint])
        _solve(problem,cp)
        if problem.status not in {"optimal",'optimal_inaccurate'}:
            rows.append({"budget":budget,"status":problem.status})
            continue
        weights=np.asarray(x.value).reshape(-1)
        lam=float(budget_constraint.dual_value)/cost_scale
        rows.append({
            "budget":float(budget),
            "loss":float(losses_np@weights),
            "cost":float(costs_np@weights),
            "lambda":lam,
            "weights":weights.tolist(),
            "kkt":_kkt(losses_np,costs_np,weights,budget,lam,float(simplex.dual_value),rtol,atol),
            "status":problem.status,
        })
    return rows


def _solve(problem,cp):
    for solver in [cp.CLARABEL,cp.HIGHS,cp.OSQP,cp.SCS]:
        try:
            problem.solve(solver=solver,verbose=False)
            if problem.status in {"optimal",'optimal_inaccurate'}:
                return
        except cp.SolverError:
            continue


def _kkt(losses,costs,x,budget,lam,mu,rtol,atol):
    active=x>1e-6
    stationarity=losses[active]+lam*costs[active]+mu
    budget_slack=costs@x-budget
    budget_tol=atol+rtol*max(abs(float(budget)),abs(float(costs@x)),1.0)
    simplex_err=abs(x.sum()-1)
    comp=float(abs(lam*budget_slack))
    stat=float(np.max(np.abs(stationarity))) if stationarity.size else 0.0
    feasible=bool(simplex_err<=atol and budget_slack<=budget_tol and np.all(x>=-atol))
    dual=bool(lam>=-atol)
    near=bool((not feasible and budget_slack<=10*budget_tol) or stat<=1e-4 or comp<=1e-4)
    status='pass' if feasible and dual and comp<=1e-5 else 'near_pass' if dual and near else 'fail'
    return {
        "primal_feasible":feasible,
        "dual_feasible":dual,
        "complementary_slackness":comp,
        "stationarity_residual":stat,
        "budget_slack":float(budget_slack),
        "budget_tolerance":float(budget_tol),
        "kkt_status":status,
    }

