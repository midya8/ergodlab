from .core import audit


def occupancy_model():
    return {'weights':[[1,1],[1,1]],'features':[[[0],[1]],[[0],[0]]],
            'transitions':[[[[3,4],[1,4]],[[1,4],[3,4]]],[[[1,4],[3,4]],[[1,4],[3,4]]]],
            'rewards':[[[[0,1],[0,1]],[[0,1],[0,1]]],[[[1,1],[1,1]],[[1,1],[1,1]]]]}


def periodic_model():
    return {'weights':[[1],[1]],'features':[[[0]],[[0]]],
            'transitions':[[[[0,1],[1,1]]],[[[1,1],[0,1]]]],
            'rewards':[[[[0,1],[0,1]]],[[[1,1],[1,1]]]]}


def sticky_model():
    m=periodic_model();m['transitions']=[[[[7,8],[1,8]]],[[[1,8],[7,8]]]];return m


def demo():
    a=audit(occupancy_model());p=audit(periodic_model());s=audit(sticky_model())
    assert a['average_reward']==[2,3] and a['gradient']['total']==[[1,18]]
    assert a['gradient']['frozen_occupancy']==[[0,1]]
    assert p['graph']['period']==2 and p['variance']['asymptotic_per_step']==[0,1]
    assert s['variance']['asymptotic_per_step']==[7,4] and s['variance']['one_step_stationary']==[1,4]
    return {'format':'ergodlab-demo-v1','occupancy':a,'periodic':p,'sticky':s,'selection':'Three prescribed finite analytical fixtures, not learned policies or benchmark trials. Sticky and alternating rewards have equal one-step stationary variance but different temporal dependence.'}
