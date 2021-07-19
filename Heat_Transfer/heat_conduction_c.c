#include<stdio.h>
#include<malloc.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>

double den[4]={300.,862.,74.2,1.18};
double c1[4]={1377.,2100.,1726.,1005.};
double kappa[4]={0.082,0.37,0.045,0.028};
//double kappa[4]={0.82,0.037,0.45,0.028};
double thi[4]={0.6,6.,3.6,5.};

int value_k(int i, double h){
	int k;
	if(i*h<thi[0]+thi[1]+thi[2]+thi[3])k=3;
	if(i*h<thi[0]+thi[1]+thi[2])k=2;
	if(i*h<thi[0]+thi[1])k=1;
	if(i*h<thi[0])k=0;
	if(fabs(i*h-thi[0])<0.0001)k=4;
	if(fabs(i*h-(thi[0]+thi[1]))<0.0001)k=5;
	if(fabs(i*h-(thi[0]+thi[1]+thi[2]))<0.0001)k=6;
	return k;
}

double* new_array1(int length){
	int i = 0;
	double *array = (double*)malloc(sizeof(double)*length);
	return array;
}

double** new_array2(int row, int col)
{
	int i = 0;
	double **array = (double**)malloc(sizeof(double*) * row);
	for (i = 0; i < row; ++i) {
		array[i] = (double*)malloc(sizeof(double) * col);
		memset(array[i], 0, sizeof(double) * col);
	}
	return array;
}

int main(){
	/*-------------------------PREPARATION-------------------------*/
	int i=0,j=0,z=0,c=0,N,M,k=0;
	//h->step;tau->time step;
	double a[4], h, tau,last_time ,tolerance;
	//initialize
	for(i=0;i<=3;i++){
		a[i]=kappa[i]/(c1[i]*den[i]);
	}
	h=0.1;tau=0.1;tolerance=1e-6,last_time=2000;
	N=(int)(15.2/h);
	M=(int)(last_time/tau);
	double *u=new_array1(N+1);
	double *v=new_array1(N+1);
	double *y=new_array1(N+1);
	double **temperature=new_array2(N+1,M);
	double *tmpup=new_array1(N+1);
	double *up=new_array1(N+1),*mid=new_array1(N+1),*down=new_array1(N+1);
	double A[7],B[7],C[7],D[4],E[4],F[4];
	double *var=new_array1(M);
	double sum=0,flag=0;
	
	for(i=0;i<4;i++){
		A[i]=-1*a[i]*1000000*tau/(2*h*h);
		B[i]=1+a[i]*1000000*tau/(h*h);
		D[i]=a[i]*1000000*tau/(2*h*h);
		C[i]=A[i];
		F[i]=D[i];
		E[i]=1-a[i]*1000000*tau/(h*h);
	}
	for(i=0;i<3;i++){                                          
		A[i+4]=1000*kappa[i]/h;
		B[i+4]=-1*1000*(kappa[i]+kappa[i+1])/h;
		C[i+4]=1000*kappa[i+1]/h;
	}
	//assignment of tridiagnal matrix
	for (z=1;z<N;z++){
		k=value_k(z,h);
		//tmp: current "up"
		tmpup[z]=C[k];
		mid[z]=B[k];
		down[z]=A[k];
		if (z==1)up[z]=tmpup[z]/mid[z];
		else up[z]=tmpup[z]/(mid[z]-down[z]*up[z-1]);
	}
	
	//initial condition
	for (i=0;i<N+1;i++){
		temperature[i][0]=38.*exp(-i*i)+37;
	}
	//boundary condition
	for (j=0;j<M;j++){
		temperature[0][j]=75.;temperature[N][j]=37.;
	}
	
	/*-------------------------CALCULATION-------------------------*/
	for(j=0;j<M-1;j++){
		for(i=0;i<N+1;i++){
			u[i]=temperature[i][j];
		}
		for(i=1;i<N;i++){
			k=value_k(i,h);
			v[i]=D[k]*u[i-1]+E[k]*u[i]+F[k]*u[i+1];
			if (k>3) v[i]=0;
		}
		v[1]=v[1]-A[0]*u[0];
		v[N-1]=v[N-1]-C[3]*u[N];
//		if(j==0)for(z=1;z<N;z++)printf("%g\n",v[z]);
		//Thomas Method
		//calculate Ly=v 
		for (z=1;z<N;z++){
			if (z==1){
				y[z]=v[z]/mid[z];
			}
			else{
				y[z]=(v[z]-down[z]*y[z-1])/(mid[z]-down[z]*up[z-1]);
			}
		}
		//calculate Uu=y 
		for (z=N-1;z>0;z--){
			if (z==N-1){
				u[z]=y[z];
			}
			else{
				u[z]=y[z]-up[z]*u[z+1];
			}
		}
		for (i=1;i<N;i++){
			temperature[i][j+1]=u[i];
		}
		sum=0;
		for(i=0;i<N+1;i++){
			sum+=(temperature[i][j+1]-temperature[i][j])*(temperature[i][j+1]-temperature[i][j]);
		}
		var[j+1]=sum;
	}		
	
	/*-------------------------WRITE-------------------------*/	
	//write temperature distribution
 	FILE *fp=NULL;
	if((fp=fopen("heat_1.txt","w+"))==NULL){
		printf("Error on open\n");
		exit(1);
	}
	for(i=0;i<N+1;i++){
		for(j=0;j<M;j++){
			if(j%50==0)fprintf(fp,"%g\t",temperature[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	//write variance with the former step
	FILE *fp2=NULL;
	if((fp2=fopen("variance.txt","w+"))==NULL){
		printf("Error on open\n");
		exit(1);
	}
	fprintf(fp2,"%10s\t %15s\t %30s","Iteration","Variance","Is smaller than tolerance\n");
	for(j=1;j<M;j++){
		if(var[j]>tolerance){
			if(j%50==0){
				fprintf(fp2,"%10d\t %15g\t %20s\n",j,var[j],"not yet");
			}
			else;
		}
		else{
			if(flag==0){
				fprintf(fp2,"%10d\t %15g\t %20s\n",j,var[j],"yes");
				break;
			}
		}
	}
	fclose(fp2);
	//not necessary
	free(u);free(v);free(y);
	free(temperature);free(tmpup);free(var);
	free(up);free(mid);free(down);
}