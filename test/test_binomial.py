def buffer_calc(Pe, n, i):
  print("--- n-i ", n-i)
  print("P0: ", Pe**i); 
  print("P2 ", (1-Pe)**(n-i)) 
  print("ret: ", (Pe**i)*((1-Pe)**(n-i)))
  return (Pe**i)*((1-Pe)**(n-i)) 


Rcrc = 0.0
Pe = 0.01 

for i in range(1,64):
    Rcrc = Rcrc + buffer_calc(Pe,64,i)

print("Rcrc: ", Rcrc)