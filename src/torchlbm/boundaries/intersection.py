import torch

def find_intersection(f_list, x1, y1, z1, x2, y2, z2, tol=1e-6, max_iter=50):
    """
    Finds intersection of a line segment with an arbitrary curve f(x, y) = 0.
    
    Args:
        f: Function defining the curve (expects x, y tensors and returns a scalar).
        x1, y1, x2, y2: Endpoints of the line segment.
        tol: Tolerance for root-finding.
        max_iter: Maximum iterations for convergence.
    
    Returns:
        (x_int, y_int): Intersection point or None if no intersection found.
    """
    # x1, y1, x2, y2 = map(torch.tensor, (x1, y1, x2, y2))
    x1, y1, z1, x2, y2, z2 = map(torch.tensor, (x1, y1, z1, x2, y2, z2))
    intersections = []

    for f in f_list:
        # print(f)
        # Function along the line segment: g(t) = f(x(t), y(t))
        def g(t):
            x_t = x1 + t * (x2 - x1)
            y_t = y1 + t * (y2 - y1)
            z_t = z1 + t * (z2 - z1)
            return f(x_t, y_t, z_t)
        
        # Check if endpoints have opposite signs (ensuring a root exists)
        g1, g2 = g(torch.tensor(0.0)), g(torch.tensor(1.0))
        # print(g1, g2)
        if g1 * g2 > 0:
            continue  # No intersection in range [0,1]
        
        # Bisection method
        t_left, t_right = torch.tensor(0.0), torch.tensor(1.0)
        for _ in range(max_iter):
            t_mid = (t_left + t_right) / 2
            g_mid = g(t_mid)
            
            if torch.abs(g_mid) < tol:
                break  # Converged
            
            if g1 * g_mid < 0:
                t_right = t_mid
            else:
                t_left = t_mid
                g1 = g_mid  # Update left function value
        
        # Compute intersection point
        t_int = (t_left + t_right) / 2
        x_int = x1 + t_int * (x2 - x1)
        y_int = y1 + t_int * (y2 - y1)
        z_int = z1 + t_int * (z2 - z1)
        
        intersections.append((t_int.item(), x_int.item(), y_int.item(), z_int.item()))

    print(f"Intersections found: {intersections}")
    if intersections:
        intersections.sort(key=lambda tup: tup[0]) 
        t_int, x_int, y_int, z_int = intersections[0]
        return t_int, x_int, y_int, z_int
    else:
        raise ValueError("No intersection found with any of the provided curves.")
    
    # # Function along the line segment: g(t) = f(x(t), y(t))
    # def g(t):
    #     x_t = x1 + t * (x2 - x1)
    #     y_t = y1 + t * (y2 - y1)
    #     z_t = z1 + t * (z2 - z1)
    #     return f(x_t, y_t, z_t)
    
    # # Check if endpoints have opposite signs (ensuring a root exists)
    # g1, g2 = g(torch.tensor(0.0)), g(torch.tensor(1.0))
    # if g1 * g2 > 0:
    #     return None  # No intersection in range [0,1]
    
    # # Bisection method
    # t_left, t_right = torch.tensor(0.0), torch.tensor(1.0)
    # for _ in range(max_iter):
    #     t_mid = (t_left + t_right) / 2
    #     g_mid = g(t_mid)
        
    #     if torch.abs(g_mid) < tol:
    #         break  # Converged
        
    #     if g1 * g_mid < 0:
    #         t_right = t_mid
    #     else:
    #         t_left = t_mid
    #         g1 = g_mid  # Update left function value
    
    # # Compute intersection point
    # t_int = (t_left + t_right) / 2
    # x_int = x1 + t_int * (x2 - x1)
    # y_int = y1 + t_int * (y2 - y1)
    # z_int = z1 + t_int * (z2 - z1)
    
    # return t_int.item(), x_int.item(), y_int.item(), z_int.item()
