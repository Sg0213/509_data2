# Chapter 3 asignment

def calculate_ndvi():
    # Find normalized vegetation difference index
    # return the value
    pass

def calculate_ndvi(nir, red):
    # Find normalized vegetation difference index
    ndvi = nir - red / nir + red   # placeholder formula (not yet correct)

    # return the value
    pass


# Assignment 1

def find_quadratic_roots(a, b, c):
    if a == 0:
        return []
    d = b*b - 4*a*c
    if d < 0:  return []
    if d == 0: return [-b/(2*a)]
    r = d**0.5
    return [(-b - r)/(2*a), (-b + r)/(2*a)]
print(find_quadratic_roots(1, -3, 2))
print(find_quadratic_roots(1,  2, 1))
print(find_quadratic_roots(1,  0, 1))





# Assignment 2
def scale_linear(x, in_min, in_max, out_min, out_max):
    m = (out_max - out_min) / (in_max - in_min)
    return out_min + m * (x - in_min)

if __name__ == "__main__":
    print(scale_linear(0.0, 0.0, 1.1, 0, 255))
    print(scale_linear(1.1, 0.0, 1.1, 0, 255))
    print(round(scale_linear(0.55, 0.0, 1.1, 0, 255))) 







