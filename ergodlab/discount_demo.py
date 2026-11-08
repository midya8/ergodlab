from .discount import audit_discounted


def curl_model():
    return {'weights':[[1,1],[1,1]],'features':[[[0,0],[1,0]],[[0,0],[0,1]]],
            'transitions':[[[[3,4],[1,4]],[[1,4],[3,4]]],[[[1,8],[7,8]],[[3,8],[5,8]]]],
            'rewards':[[[[0,1],[0,1]],[[0,1],[0,1]]],[[[1,1],[1,1]],[[1,1],[1,1]]]]}


def discount_demo():
    a=audit_discounted(curl_model());row=a['discounts'][1]
    assert a['stationary_audit']['gradient']['total']==[[1,18],[-1,18]]
    assert row['stationary_gradient_proxy']==[[1,21],[-1,21]]
    assert row['proxy_curl']==[[[0,1],[-1,147]],[[1,147],[0,1]]]
    assert row['fixed_start_normalized_objective']==[2,7] and row['fixed_start_objective_gradient']==[[5,98],[-1,98]]
    return a
