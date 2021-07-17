#include<stdio.h>
#include<string.h>
#include<math.h> 
int main(){
	
	float a1[25]={0};
    int i;
    FILE *fpRead=fopen("input_data.txt","r");  //其中"r"是表示 读
    if(fpRead==NULL)  
    {  
        return 0;  
    }  
    for( i=0;i<25;i++)  
    {  
    
        fscanf(fpRead,"%f ",&a1[i]);  
		//printf("%f\n",a1[i]);
        
    }  
	/*getchar();//等待
    fclose(fpRead);
    return 1;  */
	//for(i=0;i<24;i++){printf("%f\n",a1[i]);}
	int n[4],j;
	//printf("各层的分段数：\n");
	for(i=0; i<4; i++){
		n[i]=a1[i];
		//scanf("%d",&n[i]);
	}
	//printf("厚度："); 
	float m[4],h[4],k[4]; 
	for(i=0;i<4;i++){
		//printf("从外到里第%d 层的厚度：\n",i+1); 
		//scanf("%f",&m[i]); 
		m[i]=a1[4+i];
		h[i]=m[i]/n[i]; //每层的最小间隔 
		//printf("%f\n",m[i]);
		//printf("%f-%f-%d-%d\n",m[i],h[i],n[i],i);
		//printf("从外到里第%d 层的热传导率：\n",i+1); 
		//scanf("%f",&k[i]);
		k[i]=a1[8+i];
	}
	     
	
	int t,times;
	t=n[0]+n[1]+n[2]+n[3];
//	printf("0时刻各处的温度：\n");
	float t0,t1,a2[4],delt_t,a[t],b[t],c[t],d[t],e[t],f[t];

	//printf("初始温度（最外层温度）=\n");
//	scanf("%f",&t0);
	t0=a1[12];
//	printf("最内层温度=\n"); 
//	scanf("%f",&t1);
	t1=a1[13];
//	printf("时间阶差=\n") ;
//	scanf("%f",&delt_t);
	delt_t=a1[14];
//	printf("随时间的演化次数times=\n");
//	scanf("%d",&times);
	times=a1[15];
	float rho[4], C[4];
	for(i=0;i<4;i++){
		rho[i]=a1[16+i];
		C[i]=a1[20+i];
	}
//	printf("微分方程的系数a^2=\n") ;
//	scanf("%f",&a2);
	for(i=0;i<4;i++){
		a2[i]=k[i]/(rho[i]*C[i]);
	}
	 
	//#######################################################
	
	
//	###系数矩阵赋值 

	int r;
	for(i=0;i<=t;i++){
//	##四层的系数矩阵系数a,b,c,d,e,f
	
//	##r是用来确定层数的一个指标 
	  	if (i<=n[0]) r=0;
		if(i>n[0]&&i<=(n[1]+n[1])) r=1;
		if(i>n[1]+n[0]&&i<=(n[0]+n[1]+n[2])) r=2;
		if(i>n[2]+n[1]+n[0]&&i<=(n[0]+n[1]+n[2]+n[3])) r=3;
		
		 
		a[i]=-a2[r]*delt_t/(2*h[r]*h[r]);
		c[i]=a[i];
		b[i]=1+a2[r]*delt_t/(h[r]*h[r]);
		d[i]=a2[r]*delt_t/(2*h[r]*h[r]);
		f[i]=d[i];
		e[i]=1-a2[r]*delt_t/(h[r]*h[r]);
		
		
//		##边界处的系数
	 	if(i==n[0]){
	 		
		 	a[i]=k[r]/h[r];
			c[i]=k[r+1]/h[r+1];
			b[i]=-k[r]/h[r]-k[r+1]/h[r+1];
	}
		if(i==n[1]+n[0]){
		 	a[i]=k[r]/h[r];
			c[i]=k[r+1]/h[r+1];
			b[i]=-k[r]/h[r]-k[r+1]/h[r+1];
	}
		if(i==n[2]+n[1]+n[0]){
		 	a[i]=k[r]/h[r];
			c[i]=k[r+1]/h[r+1];
			b[i]=-k[r]/h[r]-k[r+1]/h[r+1];
	}
	printf("a%f-b%f-c%f-d%f-e%f-f%f\n",a[i],b[i],c[i],d[i],e[i],f[i]);
}
	
		
//########################################################################## 
	
//	##具体迭代运算 t是空间位置，times 可以理解为时间，也可以理解为迭代次数；
// v[t]只是一个中间代换量
	
	float u[t][times],v[t];
	
	//		## 边界条件：一个问题最内层的温度怎么处理：；
		//for(i=1;i<=3;i++){
			//这一步的目的：下面会有数组操作，为了方便计数，在数组index为负值时，数值为0；
	//	printf("%f-%d\n",m[i],n[i]);
			//m[-i]=0;
			//n[-i]=0;
		//}
		
		float key;
	
	for(i=0;i<=t+1;i++){
		//确定是第几层的
		if (i<=n[0]) {
			r=0;
			key=i*h[r];
		}
		
		if(i>n[0]&&i<=n[1]+n[0]) {
			r=1;
			key=m[0]+(i-n[0])*h[r];
		}
		
		if(i>n[1]+n[0]&&i<=n[0]+n[1]+n[2]) {
			r=2;
			key=m[1]+m[0]+(i-n[1]-n[0])*h[r];
		}
		
		if(i>n[2]+n[1]+n[0]&&i<=(t+1)) {
			r=3;
			key=m[2]+m[1]+m[0]+(i-n[2]-n[1]-n[0])*h[r];	
		}
		//没有进行迭代时，靠外面的几层的温度 服从正态分布，假设时总层数的
		//printf("%f-%d-%d-%f-%f\n",key,r,i,m[r],h[r]);
		
		 u[i][0]=75*exp(-key*key/2.26e-5);
		 if(u[i][0]<=37) u[i][0]=37;
		printf("%f$%f \n",key,u[i][0]);
	}
	
		for(i=0;i<times;i++){
				u[0][i]=t0;
				u[t][i]=t1;
				u[t+1][i]=t1; 
				printf("%d>>%f\n",i,u[t+1][i]);
			}
	

	
		for(i=0;i<=t;i++){
			f[i]=d[i];
		printf("a%f-b%f-c%f-d%f-e%f-f%f\n",a[i],b[i],c[i],d[i],e[i],f[i]);
	}
		
	
	
	
	for(j=0;j<times;j++){
				u[0][j]=t0;
				u[t][j]=t1;
				u[t+1][j]=t1; 
			//	printf("%d>>%f\n",j,u[t+1][j]);
		
	
		for(i=1;i<=t;i++){
			
//		##一个问题：N+1不知道，我们没有定义； 
			v[i]=d[i]*u[i-1][j]+e[i]*u[i][j]+d[i]*u[i+1][j];
		//if(j==0) { printf("%f\n",f[i]);
	
		// }
			
			//边界定义 
			if(i==n[0]){
				v[i]=0;
			} 
			if(i==n[1]+n[0]){
				v[i]=0;
			}
			if(i==n[0]+n[1]+n[2]){
				v[i]=0;
			}
			printf("次数%d>>v%f>d%f>e%f>f%f>%f>%f>%f\n",i,v[i],d[i],e[i],f[i],u[i-1][j],u[i][j],u[i+1][j]);
			printf("&&>%d>%f\n",i,v[i]);
//		## 这里不能有n[3]因为……一方面，上面可以对v[n[3]][0]直接定义；但是N+1未知。 
	}
			v[1]=v[1]-a[1]*u[0][j];
			v[t]=v[t]-c[t]*u[t+1][j+1]; 
//			## t+1没有定义，不知道如何？ 
			//v[1]=a[0]*v[1]/b[0] ;
//			##既然这样写，那么n[0]就不能太小，至少比2大。
	
	
//	##接下来就是最简单的具体循环运算了，这地方老师的给的解释很清楚，难度集中在如何设计变量； 
//	##消除a 
		for(i=1;i<t;i++){ 
			v[i+1]=v[i+1]-v[i]*a[i+1]/b[i]; 
			b[i+1]=b[i+1]-c[i]*a[i+1]/b[i];

		}
		
		
//	##消除c,同时主对角化为1 
			
		for(i=t;i>=1;i--){
			if (i!=1) v[i-1]=v[i-1]-c[i-1]*v[i]/b[i];
			v[i]=v[i]/b[i];
			//printf("%f \n",v[1]);
		}
			
//新旧交替；之前对V也使用了双变量，后来发现v只是一个中转，所以用一个变量（下标）减少内存浪费 
		for(i=1;i<=t;i++){
			u[i][j+1]=v[i];
		//	printf("%d>%d>%f>%f\n",j,i,u[i][0],u[i][1]); 
	}
	}
	
	
	FILE *fpWrite=fopen("output_data.txt","w");
	if(fpWrite==NULL)
	{
		return 0;
	}
	for (j=1 ; j<times ;j++){
		for (i=0; i<=t;i++){
			fprintf(fpWrite,"%f ",u[i][j]);
			}
		fprintf(fpWrite,"\n");
	}
	
		fclose(fpWrite);
	
}






