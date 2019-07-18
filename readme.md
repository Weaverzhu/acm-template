```cpp
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
// #define zerol
#define FOR(i, x, y) for (decay<decltype(y)>::type i = (x), _##i = (y); i < _##i; ++i)
#define FORD(i, x, y) for (decay<decltype(x)>::type i = (x), _##i = (y); i > _##i; --i)
#ifdef zerol
#define dbg(x...) do { cout << "\033[32;1m" << #x << " -> "; err(x); } while (0)
void err() { cout << "\033[39;0m" << endl; }
template<template<typename...> class T, typename t, typename... A>
void err(T<t> a, A... x) { for (auto v: a) cout << v << ' '; err(x...); }
template<typename T, typename... A>
void err(T a, A... x) { cout << a << ' '; err(x...); }
#else
#define dbg(...)
#endif
typedef long long LL;
typedef unsigned long long ull;
typedef long double ld;
typedef unsigned long long uLL;

const int MAXN = 1e6 + 5;
const int MAXM = 5e6 + 5;
const double eps = 1e-10;
const LL MOD = 998244353;
const LL INF = 1e18;

// =========================================

LL bin(LL a, LL b, LL p)
{
    LL res = 1;
    for (a %= p; b; b >>= 1, a = a * a % p)
        if (b & 1)
            res = res * a % p;
    return res;
}

int pr[MAXN], pcnt;

int minp[MAXN];

void init()
{
    for (int i = 2; i < MAXN; ++i)
    {
        if (!minp[i])
        {
            minp[i] = i;
            pr[pcnt++] = i;
        }
        for (int j = 0; j < pcnt; ++j)
        {
            LL nextp = 1LL * i * pr[j];
            if (nextp >= MAXN)
                break;

            if (!minp[nextp])
                minp[nextp] = pr[j];

            if (i % pr[j] == 0)
                break;
        }
    }
}

LL getphi(LL p)
{
    LL res = 1;
    for (int i = 0; i < pcnt; ++i)
    {
        if (p % pr[i] == 0)
        {
            p /= pr[i];
            res *= pr[i] - 1;
            while (p % pr[i] == 0)
            {
                p /= pr[i];
                res *= pr[i];
            }
        }
        if (1LL * i * i > p)
            break;
    }
    if (p > 1)
        res *= (p - 1);
    return res;
}

vector<LL> get_factor(LL p)
{
    vector<LL> fac;
    fac.clear();
    for (int i = 0; i < pcnt; ++i)
    {
        if (p % pr[i] == 0)
        {
            fac.push_back(pr[i]);
            while (p % pr[i] == 0)
            {
                p /= pr[i];
            }
        }
        if (1LL * i * i > p)
            break;
    }
    if (p > 1)
        fac.push_back(p);
    return fac;
}

LL find_smallest_primitive_root(LL p)
{
    if (p == 2) return 1;
    LL phi = getphi(p);
    vector<LL> fac = get_factor(phi);
    FOR(i, 2, p)
    {
        bool flag = true;
        int f_sz = (int)fac.size();
        FOR(j, 0, f_sz)
        if (bin(i, phi / fac[j], p) == 1)
        {
            flag = false;
            break;
        }
        if (flag)
            return i;
    }
    assert(0);
    return -1;
}

LL exBSGS(LL a, LL b, LL p)
{
    a %= p;
    b %= p;
    if (a == 0)
        return b > 1 ? -1 : b == 0 && p != 1;
    LL c = 0, q = 1;
    while (1)
    {
        LL g = __gcd(a, p);
        if (g == 1)
            break;
        if (b == 1)
            return c;
        if (b % g)
            return -1;
        ++c;
        b /= g;
        p /= g;
        q = a / g * q % p;
    }
    static map<LL, LL> mp;
    mp.clear();
    // LL m = sqrt(p + 1.5);
    LL m = ceil(sqrt(p))+3;
    LL v = 1;
    for (int i = 1; i <= m; ++i)
    {
        v = v * a % p;
        mp[v * b % p] = i;
    }
    for (int i = 1; i <= m; ++i)
    {
        q = q * v % p;
        auto it = mp.find(q);
        if (it != mp.end())
            return i * m - it->second + c;
    }
    return -1;
}

LL a, b, p;

LL ex_gcd(LL a, LL b, LL &x, LL &y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    LL ret = ex_gcd(b, a % b, y, x);
    y -= a / b * x;
    return ret;
}
LL get_inv(LL a, LL M)
{
    static LL x, y;
    LL res = ex_gcd(a, M, x, y);
    if (res != 1)
        return -1;
    else
        return (x % M + M) % M;
}

LL CRT(LL *m, LL *r, LL n)
{
    if (!n)
        return 0;
    LL M = m[0], R = r[0], x, y, d;
    FOR(i, 1, n)
    {
        d = ex_gcd(M, m[i], x, y);
        if ((r[i] - R) % d)
            return -1;
        x = (r[i] - R) / d * x % (m[i] / d);
        R += x * M;
        M = M / d * m[i];
        R %= M;
    }
    LL res = R >= 0 ? R : R + M;
    return res;
}

void dfs(vector<LL> &ans, LL a, LL b, int c, int d, int tans) {

    if (bin(tans, a, 1<<d) != b%(1<<d))
        return;
    if (c == d) {
        ans.push_back(tans);
        return;
    }
    dfs(ans, a, b, c, d+1, tans | (1<<d));
    dfs(ans, a, b, c, d+1, tans);
}

vector<LL> solve4(LL a, LL b, int c) {
    vector<LL> ans;
    b %= (1<<c);
    dfs(ans, a, b, c, 0, 0);
    LL p = 1<<c;
    return ans;
}

vector<LL> solve3(LL a, LL b, LL p) {
    
    a %= p; b %= p;
    vector<LL> res;
    if (b == 0) {
        LL g = p / __gcd(a, p);
        for (LL x=0; x<p; x+=g) {
            res.push_back(x);
        }
    } else {
        LL x, y;
        LL g = ex_gcd(a, p, x, y);
        if (b % g == 0) {
            LL id = b / g;
            LL newp = p / g;
            x = (x * id % newp + newp)%newp;
            for (; x<p; x+=newp)
                res.push_back(x);
        }
    }
    return res;
}


vector<LL> solve(LL a, LL b, LL p, LL phi, int cnt, LL fac)
{

    vector<LL> ans;
    if (b == 0) {
        LL g = max(1LL, cnt/a);
        g = bin(fac, g, p);
        if (g == 0) g = p;
        for (LL x=0; x<p; x+=g) {
            ans.push_back(x);
        }
        return ans;
    }
    LL g = find_smallest_primitive_root(p);
    LL s = a, t = exBSGS(g, b, p);
    if (t == -1)
        return ans;
    ans = solve3(s, t, phi);
    for (LL &x:ans) {
        // dbg(g, x, phi);
        x = bin(g, x, p);
        // dbg(x);
    }
    sort(ans.begin(), ans.end());
    ans.resize(unique(ans.begin(), ans.end()) - ans.begin());
    return ans;
}

struct Nmod
{
    LL M[50], R[50];
    vector<LL> candi[50];
    int f_sz;
    vector<LL> ans;

    void init() {
        ans.clear();
        for (int i=0; i<50; ++i) {
            candi[i].clear();
        }
        f_sz = 0;
    }

    void dfs(int now)
    {
        if (now == -1)
        {
            // dbg("FUCK");
            LL res = CRT(M, R, f_sz);
            if (res != -1)
                ans.push_back(res);
        }
        else
        {
            for (LL x : candi[now])
            {
                R[now] = x;
                dfs(now - 1);
            }
        }
    }

    vector<LL> solve2(LL a, LL b, LL p)
    {
        vector<LL> fac = get_factor(p);
        f_sz = fac.size();

        for (int i = 0; i < f_sz; ++i)
        {
            LL phi = -1;
            int cnt = 0;
            LL tp = p;
            while (tp % fac[i] == 0)
            {
                ++cnt;
                tp /= fac[i];
            }
            M[i] = bin(fac[i], cnt, INF);    
            phi = bin(fac[i], cnt - 1, INF) * (fac[i] - 1);
            if (fac[i] == 2) {
                candi[i] = solve4(a, b%M[i], cnt);
                continue;
            }
            candi[i] = solve(a, b % M[i], M[i], phi, cnt, fac[i]);
        }
        dfs(f_sz - 1);
        sort(ans.begin(), ans.end());
        unique(ans.begin(), ans.end());
        return ans;
    }

} nmod;

int main()
{
    init();
    int t;
    scanf("%d", &t);
    for (int kk = 0; kk < t; ++kk)
    {
        cin >> a >> p >> b;
        nmod.init();

        vector<LL> ans = nmod.solve2(a, b, p);
        // exit(0);
        int siz = (int)ans.size();
        if (siz == 0)
        {
            puts("No Solution");
            continue;
        }
        for (int i = 0; i < siz - 1; ++i)
            printf("%lld ", ans[i]);
        printf("%lld\n", ans[siz - 1]);
    }

    return 0;
}
```