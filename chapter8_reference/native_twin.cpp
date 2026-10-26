// Chapter 8 compact hydraulic classroom twin. SI state [x,v,P1,P2].
// RK4 with at least the requested substeps and local stiffness subdivision.
// State values are never clipped; failed trials are rejected and retried.
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {
constexpr int NP = 24;
// A1,A2,L,V01,V02,m,Ps,Pr,Patm,beta,Fc,Fs,vs,sigma,gamma,dz,Kq,Cip,
// alpha,dp_eps,makeup,relief_pressure,relief_conductance,stiffness_factor
inline double phi(double d, double eps) {
    return d / std::pow(d*d+eps*eps, .25);
}
inline double dphi(double d, double eps) {
    return (.5*d*d+eps*eps) / std::pow(d*d+eps*eps, 1.25);
}
inline double bulk(double pressure, const double* p, double bm) {
    // This floor only regularizes evaluation outside the physical domain in
    // rejected RK stages. Accepted pressure states must be strictly positive.
    double pe = std::max(pressure, 1.0);
    return 1.0 / ((1.0-p[18])/(p[9]*bm) + p[18]*p[7]/(pe*pe));
}
inline bool valid(const double* y, const double* p) {
    for(int j=0;j<4;++j) if(!std::isfinite(y[j])) return false;
    return y[2] > 0.0 && y[3] > 0.0 &&
           p[3]+p[0]*y[0] > 0.0 && p[4]+p[1]*(p[2]-y[0]) > 0.0;
}
inline double friction(double v, const double* p) {
    return (p[10]+(p[11]-p[10])*std::exp(-std::pow(v/p[12],2))) *
           std::tanh(p[14]*v) + p[13]*v;
}
inline double friction_bound(double v, const double* p) {
    double z = v/p[12], tv=std::tanh(p[14]*v);
    double mag = p[10]+(p[11]-p[10])*std::exp(-z*z);
    double dm = -2.0*(p[11]-p[10])*v/(p[12]*p[12])*std::exp(-z*z);
    return std::abs(dm*tv + mag*p[14]*(1.0-tv*tv)+p[13])/p[5];
}
inline void derivatives(const double* y, double u, double load, double bm,
                        const double* p, double* out) {
    double x=y[0],v=y[1],p1=y[2],p2=y[3];
    double ue=std::copysign(std::max(std::abs(u)-p[15],0.0),u);
    double q1=0.0,q2=0.0;
    if(ue>0.0) {
        q1=p[16]*ue*phi(p[6]-p1,p[19]);
        q2=-p[16]*ue*phi(p2-p[7],p[19]);
    } else if(ue<0.0) {
        q1=p[16]*ue*phi(p1-p[7],p[19]);
        q2=-p[16]*ue*phi(p[6]-p2,p[19]);
    }
    q1+=p[20]*std::max(p[7]-p1,0.0)-p[22]*std::max(p1-p[21],0.0);
    q2+=p[20]*std::max(p[7]-p2,0.0)-p[22]*std::max(p2-p[21],0.0);
    double leak=p[17]*(p1-p2);
    out[0]=v;
    out[1]=(p[0]*p1-p[1]*p2-p[8]*(p[0]-p[1])-friction(v,p)-load)/p[5];
    out[2]=bulk(p1,p,bm)/(p[3]+p[0]*x)*(q1-p[0]*v-leak);
    out[3]=bulk(p2,p,bm)/(p[4]+p[1]*(p[2]-x))*(q2+p[1]*v+leak);
}
inline double stable_h(const double* y,double u,double bm,const double* p) {
    double p1=y[2],p2=y[3], ue=std::max(std::abs(u)-p[15],0.0);
    double d1=0.0,d2=0.0;
    if(u>p[15]) {
        d1=p[16]*ue*dphi(p[6]-p1,p[19]);
        d2=p[16]*ue*dphi(p2-p[7],p[19]);
    } else if(u < -p[15]) {
        d1=p[16]*ue*dphi(p1-p[7],p[19]);
        d2=p[16]*ue*dphi(p[6]-p2,p[19]);
    }
    if(p1<p[7]) d1+=p[20]; if(p2<p[7]) d2+=p[20];
    if(p1>p[21]) d1+=p[22]; if(p2>p[21]) d2+=p[22];
    double vol1=p[3]+p[0]*y[0], vol2=p[4]+p[1]*(p[2]-y[0]);
    double b1=bulk(p1,p,bm),b2=bulk(p2,p,bm);
    double rate=std::max(b1/vol1*(d1+2*p[17]),b2/vol2*(d2+2*p[17]));
    rate=std::max(rate,friction_bound(y[1],p));
    rate=std::max(rate,std::sqrt((b1*p[0]*p[0]/vol1+b2*p[1]*p[1]/vol2)/p[5]));
    return p[23]/std::max(rate,1.0);
}
}
extern "C" int hydraulic_abi_version() { return 1; }
extern "C" int hydraulic_step(double* y,const double* p,int np,double u,
  double load,double bm,double dt,int substeps,int max_steps,double* diagnostics) {
    if(np!=NP || !valid(y,p) || !std::isfinite(u) || !std::isfinite(load) ||
       !std::isfinite(bm) || bm<=0 || dt<=0 || substeps<1) return -1;
    double t=0.0,maxh=dt/substeps;
    double minp=std::min(y[2],y[3]),maxp=std::max(y[2],y[3]);
    double minx=y[0],maxx=y[0];
    int steps=0,rejections=0;
    while(t < dt) {
        if(++steps>max_steps) return -2;
        double h=std::min({maxh,dt-t,stable_h(y,u,bm,p)});
        bool accepted=false;
        double k1[4],k2[4],k3[4],k4[4],a[4],b[4],c[4],next[4];
        for(int attempt=0;attempt<30;++attempt) {
            derivatives(y,u,load,bm,p,k1);
            for(int j=0;j<4;++j) a[j]=y[j]+.5*h*k1[j];
            if(valid(a,p)) {
                derivatives(a,u,load,bm,p,k2);
                for(int j=0;j<4;++j) b[j]=y[j]+.5*h*k2[j];
                if(valid(b,p)) {
                    derivatives(b,u,load,bm,p,k3);
                    for(int j=0;j<4;++j) c[j]=y[j]+h*k3[j];
                    if(valid(c,p)) {
                        derivatives(c,u,load,bm,p,k4);
                        for(int j=0;j<4;++j) next[j]=y[j]+h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;
                        if(valid(next,p)) { accepted=true; break; }
                    }
                }
            }
            h*=.5; ++rejections;
        }
        if(!accepted || h<1e-14) return -3;
        for(int j=0;j<4;++j) y[j]=next[j];
        // Diagnostics are accepted RK substep-node extrema, not a formal dense
        // continuous-time extremum certificate. High-fidelity replay is separate.
        minp=std::min({minp,y[2],y[3]}); maxp=std::max({maxp,y[2],y[3]});
        minx=std::min(minx,y[0]); maxx=std::max(maxx,y[0]);
        t+=h;
    }
    double f[4]; derivatives(y,u,load,bm,p,f);
    diagnostics[0]=f[1]; diagnostics[1]=minp;diagnostics[2]=maxp;
    diagnostics[3]=minx;diagnostics[4]=maxx;diagnostics[5]=steps;diagnostics[6]=rejections;
    return 0;
}
