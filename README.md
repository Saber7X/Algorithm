# 算法模板

# 小技巧

* 判断大小可以max或者min来简化代码和逻辑
* sqrt（x），x一定为double参数，否则会出现不可知错误，即使本地运行正确
* 数学思维题常可以通过推导方程优化求解

# 我队策略

> 开场若无法开题优先开题面短的题目（翻译快），前1-1.5小时争取写2-3道水题，前三道题由lkc和xxh承担（构造题可先推规律），前1小时的时间mwl

# 基础算法

## 二分

算法思路：假设目标值在闭区间[l, r]中， 每次将区间长度缩小一半，当l = r时，我们就找到了目标值。
### 整数二分
#### 版本1（查找符合条件区间的左端点）
>当我们将区间[l, r]划分成[l, mid]和[mid + 1, r]时，其更新操作是r = mid或者l = mid + 1;，计算mid时不需要加1。

C++ 代码模板：
```
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;//位运算，等同于（l+r）/2
        if (check(mid)) r = mid;//check函数表示条件判断正确区间是否在mid的左侧
        else l = mid + 1;
    }
    return l;
}
```
#### 版本2（查找符合条件区间区间的右端点）
>当我们将区间[l, r]划分成[l, mid - 1]和[mid, r]时，其更新操作是r = mid - 1或者l = mid;，此时为了防止死循环，计算mid时需要加1。

C++ 代码模板：
```
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;//位运算，等同于（l+r+1）/2
        if (check(mid)) l = mid;check//函数表示条件判断正确区间是否在mid的右侧
        else r = mid - 1;
    }
    return l;
}
```
### 浮点数二分
```
bool check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r)
{
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```
### STL二分函数
> 除了判断地址是否合理外，还需要判断找到的下标对应的值是否符合条件
#### lower_bound函数
>左闭右开=终点位置要比实际查找范围大1， 查找第一个大于等于val的地址（查找值下标的当前位置和右边），找不到则返回终点地址
```
lower_bound(查找范围起点,查找范围终点,查找的值val)-数组名;
```
#### upper_bound 函数
```
upper_bound(查找范围起点,查找范围终点,查找的值val)-数组名;
```
> 左闭右开=终点位置要比实际查找范围大1， 查找第一个大于val的地址（查找值下标的当前位置右边），找不到则返回终点地址

#### greater<int>()参数
> 该参数可作为二分函数的第四个参数，可求解小于等于的最大值和小的最大值，但要求数组从打到小排序
  
## 前缀和
### 一维前缀和
```
S[i] = a[1] + a[2] + ... a[i]
a[l] + ... + a[r] = S[r] - S[l - 1]
```
### 二维前缀和
```
S[i, j] = 第i行j列格子左上部分所有元素的和
以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
S[x2, y2] - S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1]

```

## 归并排序
```
void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int mid = l + r >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];

    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}

```

## 贪心
贪心算法的核心:就是局部最优到达全局最优,每一步都保证最优.

切记这里的最优,是符合题目条件的最优,往往都是题目的目标.

贪心算法没有固定的框架,但是有几大模型,从现在起我们来一步步解析.

### 区间贪心
  
# 数论

## NIM游戏
> 只能从一堆中拿至少一个石子：各堆石子数量异或和为0，后手胜，反之先手胜。
> 如果多堆石子移动一堆中的至少一个到下一堆，第一对堆的移走后消失：奇数为下标的堆的石子数异或和为0，后手胜，反之先手胜。
> sg函数：
> - 如果只有一张图，先手sg（x）=0，必败，反之必胜。
> - 如果有m张图只能选其中之一走一步，分别考虑每张图的sg，m张图的sg异或和为0，必输，反之必胜。
> 


- sg函数模板：
```cpp
int sg(int x)
{
    if(f[x]!=-1) return f[x];//f开哈希表
    set<int> S;
    for(int i=0;i<k;i++)
    {
        int sum=s[i];
        if(x>=sum) S.insert(sg(x-sum));
    }
    for(int i=0;;i++)
    {
        if(S.count(i)==0) return f[x]=i;
    }
}
```
## 求和
- 结论：
> $∑_{i=1}^ni^2=\frac {n(n+1)(2n+1)}{6}$
> \
> $∑^𝑛_{𝑖=1}𝑖^3=(\frac {𝑛(𝑛+1)} {2})^2$
> \
> $∑^𝑛_{𝑖=1}𝑖=\frac {𝑛(𝑛+1)} {2}$

## 斐波那契数列推导

> $∑^𝑛_{𝑖=1}𝑓_𝑖=𝑓_{𝑛+2}−1$
> \
>  $∑^𝑛_{𝑖=1}𝑓_{2𝑖−1}=𝑓_{2𝑛}$
>  \
> $∑^𝑛_{𝑖=1}𝑓_{2𝑖}=𝑓_{2𝑛+1}−1$
> \
> $∑^𝑛_{𝑖=1}(𝑓𝑛)^2=𝑓_𝑛𝑓_{𝑛+1}$
> \
> $𝑓_{𝑛+𝑚}=𝑓_{𝑛−1}𝑓_{𝑚−1}+𝑓_𝑛𝑓_𝑚$
> \
> $(𝑓𝑛)^2=(−1)^{(𝑛−1)}+𝑓_{𝑛−1}𝑓_{𝑛+1}$
> \
> $𝑓_{2𝑛−1}=(𝑓𝑛)^2−(𝑓_{𝑛−2})^2$
> \
> $𝑓_𝑛=\frac {𝑓_{𝑛+2}+𝑓_{𝑛−2}}3$
> \
> $\frac {𝑓_𝑖}{𝑓_{𝑖−1}}≈\frac {\sqrt{5}−1}{2}≈0.618$
> \
> $𝑓_𝑛=\frac{(\frac {1+\sqrt{5}}{2})^𝑛−(\frac{1−\sqrt{5}}{2})^𝑛}{\sqrt{5}}$

## 取模

- a mod b表示a除以b的余数。有下面的公式： 
> (a + b) % p = (a%p + b%p) %p
> \
> (a - b) % p = ((a%p - b%p) + p) %p
> \
> (a * b) % p = (a%p)*(b%p) %p
- 除法取模用逆元

## 最大公约数（GCD）和最小公倍数（LCM）

- 基本性质、定理
- - gcd(𝑎,𝑏)=gcd(𝑏,𝑎−𝑏)
- - gcd(𝑎,𝑏)=gcd(𝑏,𝑎mod𝑏)
- - gcd(𝑎,𝑏)lcm(𝑎,𝑏)=𝑎𝑏

推导结论：

- $𝑘|gcd(𝑎,𝑏)⟺𝑘|𝑎且 𝑘|𝑏$
- $gcd(𝑘,𝑎𝑏)=1⟺gcd(𝑘,𝑎)=1且 gcd(𝑘,𝑏)=1$
- (𝑎+𝑏)∣𝑎𝑏⟹gcd(𝑎,𝑏)≠1
- 在 𝐹𝑖𝑏𝑜𝑛𝑎𝑐𝑐数列中求相邻两项的 gcd时，辗转相减次数等于辗转相除次数。
- $gcd(𝑓𝑖𝑏_𝑛,𝑓𝑖𝑏_𝑚)=𝑓𝑖𝑏_{gcd(𝑛,𝑚)}$

## 裴蜀（Bézout）定理

【基本性质、定理】

-  设 𝑎,𝑏 是不全为零的整数，则存在整数 𝑥,𝑦 , 使得 $𝑎𝑥+𝑏𝑦=gcd(𝑎,𝑏)$
- $gcd(𝑎,𝑏)|𝑑⟺∃𝑥,𝑦∈Z,𝑎𝑥+𝑏𝑦=𝑑$

【推导结论】

- 设不定方程 𝑎𝑥+𝑏𝑦=gcd(𝑎,𝑏) 的一组特解为$\begin{cases} x=x_0, &  \\ y=y_0, &  \end{cases}$，则 𝑎𝑥+𝑏𝑦=𝑐 (gcd(𝑎,𝑏)|𝑐) 的通解为 $\begin{cases} 𝑥=\frac{𝑐}{gcd(𝑎,𝑏)}𝑥_0+𝑘\frac{𝑏}{gcd(𝑎,𝑏)}, &  \\ 𝑦=\frac{𝑐}{gcd(𝑎,𝑏)}𝑦_0−𝑘\frac{𝑎}{gcd(𝑎,𝑏)},  \end{cases} (𝑘∈Z)$。
- $∀𝑎,𝑏,𝑧∈N^∗,gcd(𝑎,𝑏)=1, ∃𝑥,𝑦∈N,𝑎𝑥+𝑏𝑦=𝑎𝑏−𝑎−𝑏+𝑧$


### 快速乘法取模

```cpp
long long ksc(long long a, long long b) {
    long long ans = 0;
    while(b != 0) {
	if(b & 1 != 0) {
	    ans += a;
	}
	a += a;
	b >>= 1;
    }
    return ans;
}
```

## 试除法判定质数

```cpp
int isprime(int n){
    if(n<2) return 0;
    else{
        for(int i=2;i<=n/i;i++)
        {
            if(n%i==0) return 0;
        }
    }
    return 1;
}
```

## 分解质因数

```cpp
void divide(long long int n){
    for(long long int i=2;i<=n/i;i++)
    {
        if(n%i==0){
            long long int s=0;
            while(n%i==0){
                n=n/i;
                s++;
            }
            cout<<i<<" "<<s<<endl;
        }
    }
    if(n>1) cout<<n<<" "<<"1"<<endl;
}
```

## 筛质数（筛除1～n中有多少个质数）

埃氏筛：
```cpp
int st[N];
long long int cnt=0;
long long int num[N];
int find(int n){
    for(int i=2;i<=n;i++)
    {
        if(st[i]==0)
        {
            num[cnt]=i;//存质数
            cnt++;//记质数个数
        }
        for(int j=i+i;j<=n;j+=i) st[j]=1;//把倍数筛掉
    }
}
```

线性筛：

```cpp
long long int prime[N];//存质数
bool st[N];//判断是否是质数
long long int cnt;//记质数个数
void getprime(int n){
    for(long long int i=2;i<=n;i++){
        if(st[i]==false) prime[cnt++]=i;
        for(long long int j=0;prime[j]<=n/i;j++){
            st[prime[j]*i]=true;
            if(i%prime[j]==0) break;
        }
    }
}
```

### 素数筛总结：
埃氏筛每次筛掉该次质数的所有倍数。
线性筛每次筛掉已知质数的 i 倍。

## 试除法求约数

```cpp
vector<int> getdivide(long long int n){//定义一个vector函数 返回值为vector
    vector <int> res;//存约数
    for(int i=1;i<=n/i;i++)
    {
        if(n%i==0){
            res.push_back(i);
            if(i!=n/i) res.push_back(n/i);//排除n是i的平方的情况
        }
    }
    sort(res.begin(),res.end());
    return res;
}
int main(){
    int n;
    cin>>n;
    while(n--){
        long long int a;
        cin>>a;
        vector<int> v=getdivide(a);
        for(int i=0;i<v.size();i++){//输出用v.size()
            cout<<v[i]<<" ";
        }
        cout<<endl;
    }
    return 0;
}
```
- [stl库用法](https://zhuanlan.zhihu.com/p/344558356?utm_source=qq&utm_medium=social&utm_oi=1058370565219336192)
### 算术基本定理的一些推论：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210714185215678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl81NTIzMzU3Mw==,size_16,color_FFFFFF,t_70)


## 排列
$$P^k_n=\underbrace{n\times(n-1)\times(n-2)\times\cdots(n-(k-1))}_{k个}$$

# 快速幂
```cpp
int qim(int a,int k,int p){//a^k%p的值
    int res=1;
    while(k){
        if(k & 1) res=(long long int)res*a%p;//k二进制之后的最后一位
        k >>=1;//往前移一位
        a=(long long int)a*a%p;//a更新
    }
    return res;
}
```

# 逆元
- 只有当**a与p互质**时存在逆元，是$a^{p-2}$，套一个快速幂就行了。

```cpp
#include <bits/stdc++.h>
using namespace std;

//快速幂 a^k%p
int qim(int a,int k,int p){
    int res=1;
    while(k){
        if(k & 1) res=(long long int)res*a%p;
        a=(long long int)a*a%p;
        k>>=1;
    }
    return res;
}
int main(){
    int n;
    scanf("%d",&n);
    while(n--){
        int a,p;
        scanf("%d %d",&a,&p);
        int ans=qim(a,p-2,p);
        if(a%p!=0) printf("%d\n",ans);//判断是否存在逆元
        else cout<<"impossible"<<endl;
    }
    return 0;
}
```
## 扩展欧几里德算法

- 公式a∗x+b∗y=gcd(a,b) 。 
- 若a，b互质且有解，则有a∗x+b∗y=1。 
- 当我们要求a关于b的逆元，我们可以这样看。 
- a*x % b + b*y % b = 1 % b
- a*x % b = 1 % b 
- a*x = 1 (mod b) 

```cpp
typedef long long LL;
void ex_gcd(LL a, LL b, LL &x, LL &y, LL &d){//扩展欧几里德
    if (!b) {d = a, x = 1, y = 0;}
    else{
        ex_gcd(b, a % b, y, x, d);
        y -= x * (a / b);
    }
}
LL inv(LL t, LL p){//如果不存在，返回-1
    LL d, x, y;
    ex_gcd(t, p, x, y, d);
    return d == 1 ? (x % p + p) % p : -1;
}
```

## 组合数
- $C^k_n=\frac{n!}{(n-k)!k!}$
- $C^k_n=\frac{P^k_n}{P^k_k}=\frac{\overbrace{(n-0)\times(n-1)\times(n-2)\times\cdots(n-(k-1))}^{k个}}{\underbrace{(k-0)\times(k-1)\times(k-2)\times\cdots(k-(k-1))}_{k个}}$

## 范围小的(2000)

- 通过公式打表
- 通过以下公式，类似dp打表直接输出
- $$C_a^b=C_{a-1}^{b-1}+C_{a-1}^b$$
```cpp
void init(){
    for(int i=0;i<N;i++){
        for(int j=0;j<=i;j++){
            if(j==0) num[i][j]=1;
            else num[i][j]=(num[i-1][j]+num[i-1][j-1])%mod;
        }
    }
    
}
```

## 范围稍大($10^5$)

- 快速幂和逆元
- $C_a^b=fact[a]\times infact[a-b]\times infact[b]$
- fact 为阶乘 infact为阶乘的逆元

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long LL;

const int N=100005,mod=1e9+7;
int fact[N],infact[N];

int qim(int a,int k,int p){
    int res=1;
    while(k){
        if(k&1) res=(LL)res*a%p;
        a=(LL)a*a%p;
        k>>=1;
    }
    return res;
}


int main(){
	//初始化fact和infact
    fact[0]=1;
    infact[0]=1;
    for(int i=1;i<N;i++){
        fact[i]=(LL)fact[i-1]*i%mod;
        infact[i]=(LL)infact[i-1]*qim(i,mod-2,mod)%mod;
    }
    
    int n;
    cin>>n;
    while(n--){
        int a,b;
        scanf("%d%d",&a,&b);
        printf("%lld\n",(LL)fact[a]*infact[a-b]%mod*infact[b]%mod);//注意随时取模
    }
    return 0;
}
```

## 范围超大($10^{18}$)

- Lucas定理$C_a^b≡C_{a \mod\ p}^{b \mod\ p}\times C_{a/p}^{b/p}$

```cpp
#include <bits/stdc++.h>
using namespace std;

int p;

typedef long long LL;

//快速幂求逆元
int qim(int a,int k){
    int res=1;
    while(k){
        if(k&1) res=(LL)res*a%p;
        a=(LL)a*a%p;
        k>>=1;
    }
    return res;
}

//求组合数
int c(int a,int b){
    int res=1;
    for(int i=1,j=a;i<=b;i++,j--){
        res=(LL)res*j%p;
        res=(LL)res*qim(i,p-2)%p;
    }
    return res;
}

//Lucas定理
int lucas(LL a,LL b){
    if(a<p&&b<p) return c(a,b);
    else return (LL)c(a%p,b%p)*lucas(a/p,b/p)%p;}

int main(){
    int n;
    cin>>n;
    while(n--){
        LL a,b;
        cin>>a>>b>>p;
        cout<<lucas(a,b)<<endl;
    }
    return 0;
}
```

# 不取模直接求组合数

- 先筛出a中所有的质数
- 找出a！，b！，（a-b）！中所有质数的个数
- 高精度乘法算一下


```cpp
#include <bits/stdc++.h>
#include <vector>
using namespace std;

const int N=5005;

int sum[N];//存每个素数在组合数中的个数

int prime[N],cnt;//存质数,个数
bool st[N];//判断是否是质数

//筛出n内所有的素数
void getprime(int n){
    for(long long int i=2;i<=n;i++){
        if(st[i]==false) prime[cnt++]=i;
        for(long long int j=0;prime[j]<=n/i;j++){
            st[prime[j]*i]=true;
            if(i%prime[j]==0) break;
        }
    }
}

//求n！中包含的p的个数
int get(int n,int p){
    int res=0;
    while(n){
        res+=n/p;
        n/=p;
    }
    return res;
}

//高精度乘法
vector<int> mul(vector<int> a,int b)
{
    vector<int> c;
    int t=0;
    for(int i=0;i<a.size();i++)
    {
        t+=a[i]*b;
        c.push_back(t%10);
        t/=10;
    }
    
    while(t){
        c.push_back(t%10);
        t/=10;
    }
    return c;
}

int main(){
    int a,b;
    cin>>a>>b;
    getprime(a);//筛出a内所有的素数
    //枚举所有的素数
    for(int i=0;i<cnt;i++){
        int p=prime[i];
        sum[i]=get(a,p)-get(b,p)-get(a-b,p);//算出需要做乘法的素数的个数
    }
    
    vector<int> res;
    res.push_back(1);
    
    for(int i=0;i<cnt;i++){
        for(int j=0;j<sum[i];j++){
            res=mul(res,prime[i]);
        }
    }
    for(int i=res.size()-1;i>=0;i--) printf("%d",res[i]);
    printf("\n");
    return 0;
}
```

## 一笔画问题
- 返回起点：所有顶点为偶点。
- 不返回起点：所有顶点为偶点，或有且只有两个奇点。

 
 # 组合数学
## 幻方
- 幻和：$\frac{n\times(n-1)}{2}$

## 圆排列
- 变成线排列后算出线排列的可能性再除以可以剪的位置。

## 项链排列
- 和圆排列差不多，多除一个翻转的可能。

## 多重排列
- 先看做全部不相同，算出个数n，再算出每个元素的冗余度，分别除冗余度的阶乘。

## 可重组合
- 在n个不同的元素中取r个进行组合，允许重复的组合数为：$C_{n+r-1}^r$。

## 不相邻组合
- 从1～n的n个数中取r个不相邻的数的组合数为：$C_{n-r+1}^r$。

### 组合数学小结

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210708163641307.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl81NTIzMzU3Mw==,size_16,color_FFFFFF,t_70)
- 【基本性质、定理】
- - $𝐴^𝑚_𝑛=\frac{𝑛!}{(𝑛−𝑚)!}$【排列】
- - $𝐶^𝑚_𝑛=\frac{𝐴^𝑚_𝑛}{𝑚!}=\frac{𝑛!}{𝑚!(𝑛−𝑚)!}$【组合】
- - $𝐶^𝑚_𝑛=𝐶^{𝑛−𝑚}_𝑛$【对称公式】
- - $𝐶^𝑚_𝑛=𝐶^𝑚_{𝑛−1}+𝐶^{𝑚−1}_{𝑛−1}$【加法公式】
- - $𝐶^𝑚_𝑛=\frac{𝑛}{𝑚}𝐶^{𝑚−1}_{𝑛−1}$【吸收公式】
- - $𝐶^𝑚_𝑛=(−1)^𝑚𝐶^𝑚_{𝑚−𝑛−1}$【上指标反转】
- - $∑^𝑚_{𝑖=0}𝐶^𝑖_{𝑛+𝑖}=𝐶^𝑚_{𝑛+𝑚+1}$【平移求和】
- - $∑^𝑘_{𝑖=0}𝐶^𝑖_𝑛𝐶^{𝑘−𝑖}_𝑚=𝐶^𝑘_{𝑛+𝑚}$【范德蒙德卷积】
- - $𝐶^𝑘_𝑛𝐶^𝑚_𝑘=𝐶^𝑚_𝑛𝐶^{𝑘−𝑚}_{𝑛−𝑚}$
- 【推导结论】
- - $𝑖𝑗=𝐶^2_{𝑖+𝑗}−𝐶^2_𝑖−𝐶^2_𝑗$
- - $∑^𝑛_{𝑖=0}𝐶^𝑖_{𝑛−𝑖}=𝑓𝑖𝑏_{𝑛+1}$
- - $∑^𝑛_{𝑖=0}𝐶^𝑚_𝑖=𝐶^{𝑚+1}_{𝑛+1}$（平移求和）
- - $∑^𝑛_{𝑖=0}(𝐶^𝑖_𝑛)^2=𝐶^𝑛_{2𝑛}$（范德蒙德卷积）
- - $∑^𝑛_{𝑖=0}(−1)^{𝑛−𝑖}𝐶^𝑖_𝑛𝐶^𝑚_𝑖=[𝑚=𝑛]$（可用其证明二项式反演）
- - $∑^𝑛_{𝑖=0}(−1)^{𝑖−𝑚}𝐶^𝑖_𝑛𝐶^𝑚_𝑖=[𝑚=𝑛]$（可用其证明二项式反演）
- - $∑^n_{i=0}i^2C^i_n=n(n+1)2^{n-2}$

## 欧拉函数
- 基本性质、定理
$\phi(𝑥)=𝑥∏^𝑛_𝑖=1(1−1𝑝𝑖)$,其中 𝑝𝑖 为 𝑥 的质因子，𝑛 为 𝑥 的质因子个数
$gcd(𝑎,𝑏)=1⟹\phi(𝑎𝑏)=\phi(𝑎)\phi(𝑏)$欧拉函数是积性函数）
- 推导结论
- - $𝑝>2⟹[\phi(𝑝)\mod\ 2=0]$
- - $𝑝∈ \{prime\} ⟹\phi(𝑝^𝑘)=𝑝^𝑘−𝑝^{𝑘−1}$
- - $∑^𝑛_{𝑖=1}𝑖\left[𝑔𝑐𝑑(𝑖,𝑛)=1\right]=\frac{𝑛\phi(𝑛)+[𝑛=1]}2$
- - $𝑓(𝑛)=∑^𝑛_𝑖=1\left[ gcd(𝑖,𝑘)=1\right]=\frac{𝑛}{𝑘}\phi(𝑘)+𝑓(𝑛\mod\ 𝑘)$
- 找1～n中有多少个与n互质的数
```cpp
        int a;
        cin>>a;
        int res=a;
        for(int i=2;i<=a/i;i++)
        {
            if(a%i==0)//找到质因子
            {
                res=res/i*(i-1);
                while(a%i==0) a=a/i;//把质因子除干净
            }
        }
        if(a>1) res=res/a*(a-1);//处理最后剩下的数
        cout<<res<<endl;
```


## 同余运算

- 【基本性质、定理】
- $f(n)= \begin{cases} 𝑎≡𝑏(mod𝑚) & \\ 𝑐≡𝑑(mod𝑚) &  \end{cases} ⟹𝑎+𝑐≡𝑏-𝑑(mod𝑚)$
-  $f(n)= \begin{cases} 𝑎≡𝑏(mod𝑚) & \\ 𝑐≡𝑑(mod𝑚) &  \end{cases} ⟹𝑎-𝑐≡𝑏+𝑑(mod𝑚)$
- $𝑎≡𝑏(mod\ 𝑚)⟹𝑎𝑘≡𝑏𝑘(mod\ 𝑚)$
- $𝑘𝑎≡𝑘𝑏(mod\ 𝑚),gcd(𝑘,𝑚)=1⟹𝑎≡𝑏(mod\ 𝑚)$


## 费马小定理及其扩展

- 基本性质、定理】
- - $𝑃∈\{𝑃𝑟𝑖𝑚𝑒\},𝑃∤𝑎⟹𝑎^{𝑃−1}=1(mod\ 𝑃)$
- 推导结论
- - 对于任意多项式 $𝐹(𝑥)=∑^∞_{𝑖=0}𝑎_𝑖𝑥^𝑖(a_i 对一个质数 𝑃取模），若满足 𝑎0≡1(mod\ 𝑃)，则 ∀𝑛⩽𝑃,𝐹^𝑃(𝑥)≡1(mod\ 𝑥^𝑛)$。

## 欧拉定理及其扩展
- 【基本性质、定理】
- - $gcd(𝑎,𝑚)=1⟹𝑎\phi(𝑚)≡1(mod\ 𝑚)$
- - $gcd(𝑎,𝑚)=1⟹𝑎^𝑏≡𝑎^{𝑏\mod\ \phi(𝑚)}(mod\ 𝑚)$
- - $𝑏>\phi(𝑚)⟹𝑎𝑏≡𝑎^{𝑏\ mod\ \phi(𝑚)+\phi(𝑚)}(mod\ 𝑚)$
- 【推导结论】
- - $∃𝑥∈𝑁^∗,𝑎^𝑥=1(mod\ 𝑚)⟺ gcd(𝑎,𝑚)=1$

## 孙子定理/中国剩余定理（CRT）及其扩展
- 【基本性质、定理】
- - 若$𝑚_1,𝑚_2...𝑚_𝑘$两两互素，则同余方程组 
- $\left\{ 
\begin{array}{c}
𝑥≡𝑎1(mod𝑚1) \\ 
𝑥≡𝑎2(mod𝑚2) \\ 
\vdots \\
𝑥≡𝑎𝑘(mod𝑚𝑘)\end{array}
\right.$有唯一解为：$𝑥=∑^𝑘_{𝑖=1}𝑎_𝑖𝑀_𝑖𝑀^{−1}_𝑖$,其中 $𝑀_𝑖=∏_{𝑗≠𝑖}𝑚_𝑗$。

## 勾股方程/勾股数组
- 【基本性质、定理】
- - 方程 $𝑥^2+𝑦^2=𝑧^2$ 的正整数通解为
- $\left\{ 
\begin{array}{c}
𝑥=𝑘(𝑢2−𝑣2) &\\ 
y=2kuv \\ 
z=k(u^2\ +v^2)\end{array}
\right.(𝑢,𝑣∈\{𝑃𝑟𝑖𝑚𝑒\},𝑘∈N∗),$ 且均满足 $gcd(𝑥,𝑦,𝑧)=𝑘$。

## 牛顿二项式定理
- 【基本性质、定理】
- - $(𝑥+𝑦)^𝑛=∑^𝑛_{𝑖=0}𝐶^𝑖_𝑛𝑥^{𝑛−𝑖}𝑦^𝑖$

- 【推导结论】
- - $∑^𝑛_{𝑖=0}𝐶^𝑖_𝑛=2^𝑛$
- - $∑^𝑛_{𝑖=0}𝑖𝐶^𝑖_𝑛=𝑛\times 2^{𝑛−1}$
- - $∑^𝑛_{𝑖=0}𝑖^2𝐶^𝑖_𝑛=𝑛(𝑛+1)2^{𝑛−1}$

【广义牛顿二项式定理】

- 【基本性质、定理】
- - $𝐶𝑛𝑟 = \begin{cases}
0  & \text{𝑛<0\ 𝑟∈R} \\
1 & \text{𝑛=0,𝑟∈R} \\
\frac{𝑟(𝑟−1)⋯(𝑟−𝑛+1)}{𝑛!} & \text{𝑛>0,𝑟∈R}
\end{cases}$
- - $(1+𝑥)^{−𝑛}=∑^∞_{𝑖=0}𝐶^𝑖_{−𝑛}𝑥^𝑖=∑^∞_{𝑖=0}(−1)^𝑖𝐶^𝑖_{𝑛+𝑖−1}𝑥^𝑖$
- - $(𝑥+𝑦)^\alpha=∑^∞_{𝑖=0}𝐶^𝑖_\alpha𝑥^{\alpha−𝑖}𝑦^𝑖(𝑥,𝑦,\alpha∈R 且 |𝑥𝑦|<1)$


## 卡特兰数 (Catalan)
- 基本性质、定理
- - $𝑐𝑎𝑡_𝑛=\begin{cases}1 & 𝑛=0 \\∑^{𝑛−1}_{𝑖=0}𝑐𝑎𝑡_𝑖𝑐𝑎𝑡_{𝑛−𝑖−1} & 𝑛>0\end{cases}$

【推导结论】
- - $𝑐𝑎𝑡_𝑛=𝐶^𝑛_{2𝑛}−𝐶^{𝑛+1}_{2𝑛}=\frac{𝐶^𝑛_{2𝑛}}{𝑛+1}$

## 经典容斥原理
- 推导结论
- - $𝑓(𝑖)=∑_{𝑗=𝑖}^𝑛(−1)^{𝑗−𝑖}𝐶^𝑖_𝑗𝑔(𝑗) =𝑔(𝑖)−∑_{𝑗=𝑖+1}𝐶^𝑖_𝑗𝑓(𝑗)$（𝑓(𝑖)为恰好 𝑖 个满足"balabala"的方案数，𝑔(𝑖) 为钦定 𝑖 个满足"balabala“其他随意的方案数）

## 生成函数
- 推导结论
- - 常用普通生成函数 (OGF) 收敛性式
- - - $∑^∞_{𝑖=0}𝑥^𝑖=\frac{1}{1−𝑥}$
- - - $∑^∞_{𝑖=0}𝑎^𝑖𝑥^𝑖=\frac{1}{1−𝑎𝑥}$
- - - $∑^∞_{𝑖=0}(𝑖+1)𝑥^𝑖=\frac{1}{(1−𝑥)^2}$
- - - $∑^∞_{𝑖=0}𝐶^𝑖_𝑛𝑥^𝑖=(1+𝑥)^𝑛$
- - - $∑^∞_{𝑖=0}𝐶^𝑖_{𝑛+𝑖−1}𝑥^𝑖=\frac{1}{(1−𝑥)^𝑛}$
- - - $∑^∞_{𝑖=0}𝑓𝑖𝑏_𝑖𝑥^𝑖=\frac{𝑥}{1−𝑥−𝑥^2}$（斐波那契数）
- - - $∑^∞_{𝑖=0}(∑^𝑖_{𝑗=0}𝑓𝑖𝑏_𝑗)𝑥^𝑖=\frac{𝑥}{(1−𝑥)(1−𝑥−𝑥^2)}$（斐波那契数列前缀和）
- - - $∑^∞_{𝑖=0}𝑐𝑎𝑡_𝑖𝑥^𝑖=\frac{1−\sqrt{1−4𝑥}{2𝑥}$（卡特兰数）
- - 常用指数生成函数 (EGF) 收敛性式
- - - $∑^∞_{𝑖=0}\frac{𝑥^𝑖}{𝑖!}=𝑒^𝑥$
- - - $∑^∞_{𝑖=0}\frac{𝑥^{2𝑖}{(2𝑖)!}=\frac{𝑒^𝑥+𝑒^{−𝑥}}{2}$
- - - $∑^∞_{𝑖=0}\frac{𝑥^{2𝑖+1}}{(2𝑖+1)!}=\frac{𝑒^𝑥−𝑒^{−𝑥}{2}$
- - - $∑^∞_{𝑖=0}𝐵_𝑖\frac{𝑥_𝑖}{𝑖!}=𝑒^{𝑒𝑥−1}$（贝尔数）
  
  
  
  
  
  
  
  
  
# 图论
## 图的存储

 - 图一般有两种存储方式：

1. 邻接矩阵。开个二维数组，g[i][j] 表示点 i 和点 j 之间的边权。
2. 邻接表。邻接表有两种常用写法，我推荐第二种，代码更简洁，效率也更高，后面有代码模板：
    - 二维vector：vector<vector<int>> edge，edge[i][j] 表示第 i 个点的第 j 条邻边。
- 数组模拟邻接表：为每个点开个单链表，分别存储该点的所有邻边。

#### 邻接表初始化
```cpp
void init()//初始化
{
    memset(h, -1, sizeof h)//head = -1;
    idx = 0;
}
```

#### 邻接表加边
```cpp
void add(int a,int b) //将b插入a前面 a作为根 所以处在链表的最后（头插法）
{
    e[idx]=b;
    ne[idx]=h[a];
    h[a]=idx++;
}
//无向边就添加a-b , b-a 两条边
```

#### 将坐标是k后面的点删掉
```cpp
void del(int k)
{
    ne[k] = ne[ne[k]];//使k的next指向k的下一个的下一个坐标，也就是跳过这个
}
```
#### 将x插到头结点,x为值
```
void  add_to_head(int x)
{
    e[idx] = x;//将空指针和x联系在一起
    ne[idx] = head;//将x的next坐标指向原本的头结点坐标
    head =idx;//将头坐标指向X的坐标
    idx++;//更新当前可用新坐标
}
```
#### 树和图的DFS
```
//*******基本框架
void dfs(int u)
{
    st[u] = true; // 标记一下，记录为已经被搜索过了，下面进行搜索过程
    for(int i = h[u]; i != -1; i = ne[i] )
    {
        int j = e[i];
        if(!st[j]) 
        {
            dfs(j);
        }
    }
}
//******树的重心
int h[N]; //邻接表存储树，有n个节点，所以需要n个队列头节点
int e[M]; //存储元素
int ne[M]; //存储列表的next值
int idx; //单链表指针
int n; //题目所给的输入，n个节点
int ans = N; //表示重心的所有的子树中，最大的子树的结点数目
bool st[N]; //记录节点是否被访问过，访问过则标记为true

//a所对应的单链表中插入b  a作为根 
void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}
int dfs(int u) {
    int res = 0; //存储 删掉某个节点之后，最大的连通子图节点数
    st[u] = true; //标记访问过u节点
    int sum = 1; //存储 以u为根的树 的节点数, 包括u，如图中的4号节点

    //访问u的每个子节点
    for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        //因为每个节点的编号都是不一样的，所以 用编号为下标 来标记是否被访问过
        if (!st[j]) {
            int s = dfs(j);  // u节点的单棵子树节点数 如图中的size值
            res = max(res, s); // 记录最大联通子图的节点数
            sum += s; //以j为根的树 的节点数
        }
    }
    
    //n-sum 如图中的n-size值，不包括根节点4；
    res = max(res, n - sum); // 选择u节点为重心，最大的 连通子图节点数
    ans = min(res, ans); //遍历过的假设重心中，最小的最大联通子图的 节点数
    return sum;
}
```
#### 树和图的BFS
```cpp
int n, m;
int h[N], e[M], ne[M], idx;
int d[N];

queue<int> q;

void add(int a, int b)  // 添加一条边a->b
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void bfs()
{
    q.push(1);
    d[1] = 0;
    while (!q.empty())
    {
        int t = q.front();
        q.pop();
        for (int i = h[t]; i != -1; i = ne[i] )
        {
            int j = e[i];
            if (d[j] == -1)
            {
                d[j] = d[t] + 1;
                q.push(e[i]);
            }
        }
    }
}
```
```cpp
int bfs()
{
	memset(d, -1, sizeof d);
	q[0] = 1;//把第一个点放进队列 
	int hh = 0, tt = 0;
	d[1] = 0;//第一个点到起点的距离为0 
	
	while(hh <= tt)//手动模拟队列 
	{
		int t = q[hh ++] ;//取出队头 
		
		for(int i = h[t]; i != -1; i = ne[i])//模板 
		{
			int j = e[i];
			
			if(d[j] == -1)//如果没有走过 
			{
				d[j] = d[t] + 1;//更新当前点到起点的距离 
				
				q[++tt] = j;//放入队列 
			}
		}
	}
	return d[n];
}
```

## 最短路
- 最短路算法分为两大类：

1. 单源最短路，常用算法有：
**(1)** **dijkstra**，只有所有边的权值为正时才可以使用。在稠密图上的时间复杂度是 O(n2)，稀疏图上的时间复杂度是 O(mlogn)。
**(2)** **spfa**，不论边权是正的还是负的，都可以做。算法平均时间复杂度是 O(km)，k 是常数。 强烈推荐该算法。
2. 多源最短路，一般用floyd算法。代码很短，三重循环，时间复杂度是 O(n3)。

## 题目大意
给一张无向图， n 个点 m 条边，求从1号点到 n 号点的最短路径。
输入中可能包含重边。

### dijkstra算法 O(n2)

最裸的dijkstra算法，不用堆优化。每次暴力循环找距离最近的点。
只能处理边权为正数的问题。
图用邻接矩阵存储。
```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010, M = 2000010, INF = 1000000000;

int n, m;
int g[N][N], dist[N];   // g[][]存储图的邻接矩阵, dist[]表示每个点到起点的距离
bool st[N];     // 存储每个点的最短距离是否已确定

void dijkstra()
{
    for (int i = 1; i <= n; i++) dist[i] = INF;
    dist[1] = 0;
    for (int i = 0; i < n; i++)
    {
        int id, mind = INF;
        for (int j = 1; j <= n; j++)
            if (!st[j] && dist[j] < mind)
            {
                mind = dist[j];
                id = j;
            }
        st[id] = 1;
        for (int j = 1; j <= n; j++) dist[j] = min(dist[j], dist[id] + g[id][j]);
    }
}

int main()
{
    cin >> m >> n;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            g[i][j] = INF;
    for (int i = 0; i < m; i++)
    {
        int a, b, c;
        cin >> a >> b >> c;
        g[a][b] = g[b][a] = min(g[a][b], c);
    }
    dijkstra();
    cout << dist[n] << endl;
    return 0;
}
```


### dijkstra+heap优化 O(mlogn)
用堆维护所有点到起点的距离。时间复杂度是 O(mlogn)。
这里我们可以手写堆，可以支持对堆中元素的修改操作，堆中元素个数不会超过 n。也可以直接使用STL中的priority_queue，但不能支持对堆中元素的修改，不过我们可以将所有修改过的点直接插入堆中，堆中会有重复元素，但堆中元素总数不会大于 m。
只能处理边权为正数的问题。
图用邻接表存储。

```cpp
typedef pair<int, int> PII;

int n;      // 点的数量
int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
int dist[N];        // 存储所有点到1号点的距离
bool st[N];     // 存储每个点的最短距离是否已确定

// 求1号点到n号点的最短距离，如果不存在，则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});      // first存储距离，second存储节点编号

    while (heap.size())
    {
        auto t = heap.top();
        heap.pop();

        int ver = t.second, distance = t.first;

        if (st[ver]) continue;
        st[ver] = true;

        for (int i = h[ver]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > distance + w[i])
            {
                dist[j] = distance + w[i];
                heap.push({dist[j], j});
            }
        }
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```

### spfa算法 O(km)
bellman-ford算法的优化版本，可以处理存在负边权的最短路问题。
最坏情况下的时间复杂度是 O(nm)，但实践证明spfa算法的运行效率非常高，期望运行时间是 O(km)，其中 k 是常数。
但需要注意的是，在网格图中，spfa算法的效率比较低，如果边权为正，则尽量使用 dijkstra 算法。

图采用邻接表存储。
队列为手写的循环队列。

```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 1010, M = 2000010, INF = 1000000000;

int n, m;
int dist[N], q[N];      // dist表示每个点到起点的距离, q 是队列
int h[N], e[M], v[M], ne[M], idx;       // 邻接表
bool st[N];     // 存储每个点是否在队列中

void add(int a, int b, int c)
{
    e[idx] = b, v[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

void spfa()
{
    int hh = 0, tt = 0;
    for (int i = 1; i <= n; i++) dist[i] = INF;
    dist[1] = 0;
    q[tt++] = 1, st[1] = 1;
    while (hh != tt)
    {
        int t = q[hh++];
        st[t] = 0;
        if (hh == n) hh = 0;
        for (int i = h[t]; i != -1; i = ne[i])
            if (dist[e[i]] > dist[t] + v[i])
            {
                dist[e[i]] = dist[t] + v[i];
                if (!st[e[i]])
                {
                    st[e[i]] = 1;
                    q[tt++] = e[i];
                    if (tt == n) tt = 0;
                }
            }
    }
}

int main()
{
    memset(h, -1, sizeof h);
    cin >> m >> n;
    for (int i = 0; i < m; i++)
    {
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c);
        add(b, a, c);
    }
    spfa();
    cout << dist[n] << endl;
    return 0;
}
```

### floyd算法 O(n3)
标准弗洛伊德算法，三重循环。循环结束之后 d[i][j] 存储的就是点 i 到点 j 的最短距离。
需要注意循环顺序不能变：第一层枚举中间点，第二层和第三层枚举起点和终点。

由于这道题目的数据范围较大，点数最多有1000个，因此floyd算法会超时。
但我们的目的是给出算法模板哦~

```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 1010, M = 2000010, INF = 1000000000;

int n, m;
int d[N][N];    // 存储两点之间的最短距离

int main()
{
    cin >> m >> n;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            d[i][j] = i == j ? 0 : INF;
    for (int i = 0; i < m; i++)
    {
        int a, b, c;
        cin >> a >> b >> c;
        d[a][b] = d[b][a] = min(c, d[a][b]);
    }
    // floyd 算法核心
    for (int k = 1; k <= n; k++)
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
    cout << d[1][n] << endl;
    return 0;
}
```
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

# STL库函数

## vector
>vector是变长数组，支持随机访问，不支持在任意位置O(1)插入。为了保证效率，元素的增删一般应该在末尾进行。
```
#include <vector> 	//头文件
vector<int> a;		//相当于一个长度动态变化的int数组
vector<int> b[233];	//相当于第一维长233，第二位长度动态变化的int数组
struct rec{…};
vector<rec> c;		                                    //自定义的结构体类型也可以保存在vector中
```
* size/empty

 * size函数返回vector的实际长度（包含的元素个数）
 * empty函数返回一个bool类型，表明vector是否为空。二者的时间复杂度都是O(1)。
* clear:clear函数把vector清空。

> 迭代器

* 迭代器就像STL容器的“指针”，可以用星号“*”操作符解除引用。
* 一个保存int的vector的迭代器声明方法为：vector<int>::iterator it;
* vector的迭代器是“随机访问迭代器”，可以把vector的迭代器与一个整数相加减，其行为和指针的移动类似。
* 可以把vector的两个迭代器相减，其结果也和指针相减类似，得到两个迭代器对应下标之间的距离。

* begin/end
 * begin函数返回指向vector中第一个元素的迭代器。例如a是一个非空的vector，则*a.begin()与a[0]的作用相同。
 * 所有的容器都可以视作一个“前闭后开”的结构，end函数返回vector的尾部，即第n个元素再往后的“边界”。*a.end()与a[n]都是越界访问，其中n=a.size()。
 * 下面两份代码都遍历了vector<int>a，并输出它的所有元素。
```
for (int I = 0; I < a.size(); I ++) cout << a[i] << endl;
for (vector<int>::iterator it = a.begin(); it != a.end(); it ++) cout << *it << endl;
```
* front/back
 *		front函数返回vector的第一个元素，等价于*a.begin() 和 a[0]。
 *		back函数返回vector的最后一个元素，等价于*==a.end() 和 a[a.size() – 1]。

* push_back() 和 pop_back()
 * a.push_back(x) 把元素x插入到vector a的尾部。
 * b.pop_back() 删除vector a的最后一个元素。
## pair
```
pair<t1,t2> a;		//定义一个对组a,对组的有两个值，t1,t2为数据类型，相当于定义一个结构体，结构体中有两个值
a.first//返回pair中的第一个值
a.second//返回pair中的第二个值
```
>make_pair(first,last)：由传递给它的两个实参生成一个新的pair对象

在vector中的pair的使用

声明vector：

vector<pair<int,int> >vec

往vector中插入数据，需要用到make_pair:

vec.push_back(make_pair(20,30));   

vec.push_back(make_pair<int,int>(10,50));


定义迭代器：

 vector<pair<int,int> > ::iterator iter;

for(iter=vec.begin();iter!=vec.end();iter++);

数据读取：

第一个数据:(*iter).first

第二个数据:(*iter).second

##map
map <下标数据类型，存的数的数据类型>  数组名

例：

    map <string , int> num;
        num["abc"]=1;


# 数据结构

## 树状数组
```
#include<bits/stdc++.h>
using namespace std;
const int N=100005;
int n,m;
int tr[N];
int lowbit(int x)
{
    return x & -x;
}
void add(int x,int v)
{
    for(int i=x;i<=n;i+=lowbit(i))
    {
        tr[i]+=v;
    }
}
int query(int x)
{
    int res=0;
    for(int i=x;i>0;i-=lowbit(i))
    {
        res+=tr[i];
    }
    return res;
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        int num;
        cin>>num;
        add(i,num);
    }
    while(m--)
    {
        int k,a,b;
        cin>>k>>a>>b;
        if(k==1)
        {
            add(a,b);
        }else{
            cout<<query(b)-query(a-1)<<endl;
        }
    }
    return 0;
}
```

# 并查集

### 朴素
```
int p[N]; //存储每个点的祖宗节点

// 返回x的祖宗节点
int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

// 初始化，假定节点编号是1~n
for (int i = 1; i <= n; i ++ ) p[i] = i;

// 合并a和b所在的两个集合：
p[find(a)] = find(b);
```

### 维护size
```
int p[N], size[N];
    //p[]存储每个点的祖宗节点, size[]只有祖宗节点的有意义，表示祖宗节点所在集合中的点的数量

    // 返回x的祖宗节点
    int find(int x)
    {
        if (p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    // 初始化，假定节点编号是1~n
    for (int i = 1; i <= n; i ++ )
    {
        p[i] = i;
        size[i] = 1;
    }

    // 合并a和b所在的两个集合：
    size[find(b)] += size[find(a)];
    p[find(a)] = find(b);
```

### 维护到祖宗节点距离
```
int p[N], d[N];
//p[]存储每个点的祖宗节点, d[x]存储x到p[x]的距离

    // 返回x的祖宗节点
    int find(int x)
    {
        if (p[x] != x)
        {
            int u = find(p[x]);
            d[x] += d[p[x]];
            p[x] = u;
        }
        return p[x];
    }

    // 初始化，假定节点编号是1~n
    for (int i = 1; i <= n; i ++ )
    {
        p[i] = i;
        d[i] = 0;
    }

    // 合并a和b所在的两个集合：
    p[find(a)] = find(b);
    d[find(a)] = distance; //  根据具体问题，初始化find(a)的偏移量
```


# 递归
## （1）指数型枚举
从 1∼n 这 n 个整数中随机选取任意多个，输出所有可能的选择方案。

输入格式
输入一个整数 n。

输出格式
每行输出一种方案。

同一行内的数必须升序排列，相邻两个数用恰好 1 个空格隔开。

对于没有选任何数的方案，输出空行。

本题有自定义校验器（SPJ），各行（不同方案）之间的顺序任意。

**数据范围**
1≤n≤15

**输入样例：**
3

**输出样例：**

3
2
2 3
1
1 3
1 2
1 2 3
```
#include<bits/stdc++.h>
using namespace std;
int n;
int res[100]={0};//bool类型只有0和1，所以用int记录状态，0待选，1已选，2不选 
void dfs(int u)
{
	if(u>n)//到达边界
	{
		for(int i=1;i<=n;i++)
		{
			if(res[i]==1)
			{
				cout<<i<<" ";
			}
		}
		cout<<endl;
		return;返回上层
	}
	res[u]=2;//不选
	dfs(u+1);
	res[u]=0;//恢复状态
	
	res[u]=1;//选
	dfs(u+1);
	res[u]=0;//恢复状态
}
int main()
{
	cin>>n;
	dfs(1);
	return 0;
}

```
## （2）组合型枚举
从 1∼n 这 n 个整数中随机选出 m 个，输出所有可能的选择方案。

输入格式
两个整数 n,m ,在同一行用空格隔开。

输出格式
按照从小到大的顺序输出所有方案，每行 1 个。

首先，同一行内的数升序排列，相邻两个数用一个空格隔开。

其次，对于两个不同的行，对应下标的数一一比较，字典序较小的排在前面（例如 1 3 5 7 排在 1 3 6 8 前面）。

**数据范围**
n>0 ,
0≤m≤n ,
n+(n−m)≤25

**输入样例：**
5 3

**输出样例：**
1 2 3 
1 2 4 
1 2 5 
1 3 4 
1 3 5 
1 4 5 
2 3 4 
2 3 5 
2 4 5 
3 4 5 
```
#include<cstring>
#include<algorithm>
#include<iostream>
using namespace std;
int n,m;
bool res[100]={0};
int ans[100];
void dfs(int u)
{
	if(u>m)//达到边界条件
	{
		for(int i=1;i<=m;i++)
		{
			cout<<ans[i]<<" ";
		}
		cout<<endl;
		return;
	}
	for(int i=ans[u-1];i<=n;i++)//从上一个开始枚举
	{
		if(res[i]==0&&i!=0)//如果当前元素没有被选过
		{
			ans[u]=i;
			res[i]=1;//标记为选过
			dfs(u+1);
			res[i]=0;//恢复状态
		}
	}
}

int main()
{
	cin>>n>>m;
	dfs(1);
	return 0;
}
```
##（3）排列型枚举
把 1∼n 这 n 个整数排成一行后随机打乱顺序，输出所有可能的次序。

**输入格式**
一个整数 n。

**输出格式**
按照从小到大的顺序输出所有方案，每行 1 个。

首先，同一行相邻两个数用一个空格隔开。

其次，对于两个不同的行，对应下标的数一一比较，字典序较小的排在前面。

**数据范围**
1≤n≤9

**输入样例：**
3

**输出样例：**
1 2 3
1 3 2
2 1 3
2 3 1
3 1 2
3 2 1

```
#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;
int n;
bool res[100]={0};
int ans[100];
void dfs(int u)
{
	if(u>n)
	{
		for (int i=1;i<=n;i++)
		{
			cout<<ans[i]<<" ";
		}
		cout<<endl;
		return;
	}
	for(int i=1;i<=n;i++)
	{
		if(res[i]==0)//如果没有被选过
		{
			ans[u]=i;
			res[i]=1;//标记为选过
			dfs(u+1);
			res[i]=0;//恢复状态
		}
	} 
}

int main()
{
	cin>>n;
	dfs(1);
	return 0;
}
```

## （4）递归求斐波那契
请使用递归的方式求斐波那契数列的第 n 项。

斐波那契数列：1,1,2,3,5…，这个数列从第 3 项开始，每一项都等于前两项之和

**输入格式**
共一行，包含整数 n。

**输出格式**
共一行，包含一个整数，表示斐波那契数列的第 n 项。

**数据范围**
1≤n≤30

**输入样例：**
4

**输出样例：**
3

```
#include<bits/stdc++.h>
using namespace std;
int n;
int dfs(int u)
{
	if(u<=0)
	{
		return 1;
	}
	return dfs(u-1)+dfs(u-2);
}
int main()
{
	
	cin>>n;
	cout<<dfs(n);
	return 0;
 } 
```

# 动态规划
## （1）背包DP
#### ① 01背包(只有一件物品)
```
#include<iostream> 
using namespace std;
int main()
{
	int n,m;
	int f[1005]={0};
	cin>>n>>m;
	int w;
	int v;
	for(int i=1;i<=n;i++)
	{
		cin>>v>>w;
		for(int j=m;j>=v;j--)
		{
			f[j]=max(f[j],f[j-v]+w);
		}
	}
	cout<<f[m];
	return 0;
}
```

#### ② 完全背包（每种物品有无限件）
```
#include<bits/stdc++.h> 
using namespace std;
const int N=1005;

int main()
{
	int n,m;
	int f[N]={0};
	int v[N];
	int w[N];
	cin>>n>>m;
	for(int i=1;i<=n;i++)
	{
		cin>>v[i]>>w[i];
	}
	for(int i=1;i<=n;i++)
	{
		for(int j=0;j<=m;j++)
		{
			if(j>=v[i])
			{
				f[j]=max(f[j],f[j-v[i]]+w[i]);
			}
			
		}
	}
	cout<<f[m];
	return 0;
}

```
#### ③ 多重背包（每种物品有限定件数）
```
#include<bits/stdc++.h>
using namespace std;
const int N=105;
int main() 
{
	int n,m;
	cin>>n>>m;
	int v[N],w[N],s[N];//体积，价值，重量
	int f[N][N]={0};
	for(int i=1;i<=n;i++) 
	{
		cin>>v[i]>>w[i]>>s[i];
	}
	for(int i=1;i<=n;i++)
	{
		for(int j=0;j<=m;j++)
		{
			for(int k=0;k<=s[i];k++)
			{
				if(k*v[i]<=j)
				{
					f[i][j]=max(f[i][j],f[i-1][j-k*v[i]]+w[i]*k);
				}
				
			}
		}
	}
	cout<<f[n][m];
	return 0;
}

```
#### ④ 二维费用背包（有三个限制条件）
```
#include<bits/stdc++.h>
using namespace std;
int main() 
{
	int n,V,M;
	int v,w,m;
	cin>>n>>V>>M;
	int f[105][105]={0};
	for(int i=1;i<=n;i++)
	{
		//体积，重量，价值 
		cin>>v>>m>>w;
		for(int j=V;j>=v;j--)
		{
			for(int k=M;k>=m;k--)
			{
				f[j][k]=max(f[j][k],f[j-v][k-m]+w);
			}
		}
	}
	cout<<f[V][M];
	return 0;
}

```
##	（2）线性DP
####① 数字三角形
给定一个如下图所示的数字三角形，从顶部出发，在每一结点可以选择移动至其左下方的结点或移动至其右下方的结点，一直走到底层，要求找出一条路径，使路径上的数字的和最大。

                7
              3   8
            8   1   0
          2   7   4   4
        4   5   2   6   5

**输入格式**
第一行包含整数 n，表示数字三角形的层数。

接下来 n 行，每行包含若干整数，其中第 i 行表示数字三角形第 i 层包含的整数。

**输出格式**
输出一个整数，表示最大的路径数字和。

**数据范围**
1≤n≤500,
−10000≤三角形中的整数≤10000

**输入样例：**
5
7
3 8
8 1 0 
2 7 4 4
4 5 2 6 5

**输出样例：**
30

```
#include<bits/stdc++.h>
using namespace std;
const int N=510,inf=0x3f3f3f3f;
int f[N][N]={-inf,-inf}, a[N][N], n;

int main()
{
	cin>>n;
	for(int i=1;i<=n;i++)//读入三角形
	{
		for(int j=1;j<=i;j++)
		{
			cin>>a[i][j];
		}
	}
	
	for(int i=1;i<=n;i++)
	{
		for(int j=0;j<=n;j++)
		{
			f[i][j]=-inf;
		}
	}
	
	f[1][1]=a[1][1];
	for(int i=2;i<=n;i++)
	{
		for(int j=1;j<=i;j++)
		{
			f[i][j]=a[i][j]+max(f[i-1][j-1],f[i-1][j]);//每一层都选择上面左右两个中最大的那个并于当前位置相加
		}
	}
	
	int maxn=-inf;
	for(int i=1;i<=n;i++)//找最后一行中最大的数
	{
		maxn=max(maxn,f[n][i]);
	}
	cout<<maxn;
	return 0;
}
```
#### ② 走方格
**题目：**
在平面上有一些二维的点阵。
这些点的编号就像二维数组的编号一样，从上到下依次为第 11 至第 nn 行，从左到右依次为第 11 至第 mm 列，每一个点可以用行号和列号来表示。
现在有个人站在第 11 行第 11 列，要走到第 nn 行第 mm 列。
只能向右或者向下走。
注意，如果行号和列数都是偶数，不能走入这一格中。
问有多少种方案。

**输入格式**
输入一行包含两个整数 n,m。

**输出格式**
输出一个整数，表示答案。

**数据范围**
1≤n,m≤301≤n,m≤30

**输入样例1：**
3 4

**输出样例1：**
2

**输入样例2：**
6 6

**输出样例2：**
0
```
#include<iostream>
int main()
{
	int n,m; 
int f[105][105];
	scanf("%d %d",&n,&m);
	for(int i=1;i<=n;i++)
	{
		f[i][1]=1;
	}
	for(int j=1;j<=m;j++)
	{
		f[1][j]=1;
	}
	for(int i=2;i<=n;i++)
	{
		for(int j=2;j<=m;j++)
		{
			if(i%2!=0||j%2!=0) f[i][j]=f[i-1][j]+f[i][j-1];//和上面的和左边的相加
		}
	}
	if(n%2==0&&m%2==0)
	{
	     f[n][m]=0;
	}
	printf("%d",f[n][m]);
	return 0;
 }

```
# 高精度
## 高精度加法
```
// C = A + B, A >= 0, B >= 0
vector<int> add(vector<int> &A, vector<int> &B)
{
    if (A.size() < B.size()) return add(B, A);

    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++ )
    {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }

    if (t) C.push_back(t);
    return C;
}
```

## 高精度减法
```
// C = A - B, 满足A >= B, A >= 0, B >= 0
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++ )
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

## 高精度乘低精度
```
// C = A * b, A >= 0, b >= 0
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C;

    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();

    return C;
}
```

## 高精度除以低精度
```
// A / b = C ... r, A >= 0, b > 0
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

