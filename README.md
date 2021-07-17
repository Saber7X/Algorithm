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
  
  
  
  
  
  
  
  
  
  
  
# 图论
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

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
