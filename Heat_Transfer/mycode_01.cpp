#include<stdio.h>
#include<string.h>
#include<math.h> 
int main(){
	
	float a1[25]={0};
    int i;
    FILE *fpRead=fopen("input_data.txt","r");  //����"r"�Ǳ�ʾ ��
    if(fpRead==NULL)  
    {  
        return 0;  
    }  
    for( i=0;i<25;i++)  
    {  
    
        fscanf(fpRead,"%f ",&a1[i]);  
		//printf("%f\n",a1[i]);
        
    }  
	/*getchar();//�ȴ�
    fclose(fpRead);
    return 1;  */
	//for(i=0;i<24;i++){printf("%f\n",a1[i]);}
	int n[4],j;
	//printf("����ķֶ�����\n");
	for(i=0; i<4; i++){
		n[i]=a1[i];
		//scanf("%d",&n[i]);
	}
	//printf("��ȣ�"); 
	float m[4],h[4],k[4]; 
	for(i=0;i<4;i++){
		//printf("���⵽���%d ��ĺ�ȣ�\n",i+1); 
		//scanf("%f",&m[i]); 
		m[i]=a1[4+i];
		h[i]=m[i]/n[i]; //ÿ�����С��� 
		//printf("%f\n",m[i]);
		//printf("%f-%f-%d-%d\n",m[i],h[i],n[i],i);
		//printf("���⵽���%d ����ȴ����ʣ�\n",i+1); 
		//scanf("%f",&k[i]);
		k[i]=a1[8+i];
	}
	     
	
	int t,times;
	t=n[0]+n[1]+n[2]+n[3];
//	printf("0ʱ�̸������¶ȣ�\n");
	float t0,t1,a2[4],delt_t,a[t],b[t],c[t],d[t],e[t],f[t];

	//printf("��ʼ�¶ȣ�������¶ȣ�=\n");
//	scanf("%f",&t0);
	t0=a1[12];
//	printf("���ڲ��¶�=\n"); 
//	scanf("%f",&t1);
	t1=a1[13];
//	printf("ʱ��ײ�=\n") ;
//	scanf("%f",&delt_t);
	delt_t=a1[14];
//	printf("��ʱ����ݻ�����times=\n");
//	scanf("%d",&times);
	times=a1[15];
	float rho[4], C[4];
	for(i=0;i<4;i++){
		rho[i]=a1[16+i];
		C[i]=a1[20+i];
	}
//	printf("΢�ַ��̵�ϵ��a^2=\n") ;
//	scanf("%f",&a2);
	for(i=0;i<4;i++){
		a2[i]=k[i]/(rho[i]*C[i]);
	}
	 
	//#######################################################
	
	
//	###ϵ������ֵ 

	int r;
	for(i=0;i<=t;i++){
//	##�Ĳ��ϵ������ϵ��a,b,c,d,e,f
	
//	##r������ȷ��������һ��ָ�� 
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
		
		
//		##�߽紦��ϵ��
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
	
//	##����������� t�ǿռ�λ�ã�times �������Ϊʱ�䣬Ҳ�������Ϊ����������
// v[t]ֻ��һ���м������
	
	float u[t][times],v[t];
	
	//		## �߽�������һ���������ڲ���¶���ô������
		//for(i=1;i<=3;i++){
			//��һ����Ŀ�ģ�����������������Ϊ�˷��������������indexΪ��ֵʱ����ֵΪ0��
	//	printf("%f-%d\n",m[i],n[i]);
			//m[-i]=0;
			//n[-i]=0;
		//}
		
		float key;
	
	for(i=0;i<=t+1;i++){
		//ȷ���ǵڼ����
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
		//û�н��е���ʱ��������ļ�����¶� ������̬�ֲ�������ʱ�ܲ�����
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
			
//		##һ�����⣺N+1��֪��������û�ж��壻 
			v[i]=d[i]*u[i-1][j]+e[i]*u[i][j]+d[i]*u[i+1][j];
		//if(j==0) { printf("%f\n",f[i]);
	
		// }
			
			//�߽綨�� 
			if(i==n[0]){
				v[i]=0;
			} 
			if(i==n[1]+n[0]){
				v[i]=0;
			}
			if(i==n[0]+n[1]+n[2]){
				v[i]=0;
			}
			printf("����%d>>v%f>d%f>e%f>f%f>%f>%f>%f\n",i,v[i],d[i],e[i],f[i],u[i-1][j],u[i][j],u[i+1][j]);
			printf("&&>%d>%f\n",i,v[i]);
//		## ���ﲻ����n[3]��Ϊ����һ���棬������Զ�v[n[3]][0]ֱ�Ӷ��壻����N+1δ֪�� 
	}
			v[1]=v[1]-a[1]*u[0][j];
			v[t]=v[t]-c[t]*u[t+1][j+1]; 
//			## t+1û�ж��壬��֪����Σ� 
			//v[1]=a[0]*v[1]/b[0] ;
//			##��Ȼ����д����ôn[0]�Ͳ���̫С�����ٱ�2��
	
	
//	##������������򵥵ľ���ѭ�������ˣ���ط���ʦ�ĸ��Ľ��ͺ�������Ѷȼ����������Ʊ����� 
//	##����a 
		for(i=1;i<t;i++){ 
			v[i+1]=v[i+1]-v[i]*a[i+1]/b[i]; 
			b[i+1]=b[i+1]-c[i]*a[i+1]/b[i];

		}
		
		
//	##����c,ͬʱ���Խǻ�Ϊ1 
			
		for(i=t;i>=1;i--){
			if (i!=1) v[i-1]=v[i-1]-c[i-1]*v[i]/b[i];
			v[i]=v[i]/b[i];
			//printf("%f \n",v[1]);
		}
			
//�¾ɽ��棻֮ǰ��VҲʹ����˫��������������vֻ��һ����ת��������һ���������±꣩�����ڴ��˷� 
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






