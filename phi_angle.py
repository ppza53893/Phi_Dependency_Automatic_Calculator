

cont = 1
while cont == 1:
    
    print("Input angle of λ/2 plane")
    phi_plane = float(input())
    
    for i in range(9):

        if phi_plane + 11.25 * i < 360:
            
            print("φ_p = ", 22.5 * i, ", φ_λ/2 = ", phi_plane + 11.25 * i)

        else:

            print("φ_p = ", 22.5 * i, ", φ_λ/2 = ", phi_plane + 11.25 * i, "(",phi_plane + 11.25 * i - 360 , ")")
    

    print("Continue? (yes = 1 / no = 0)")
    cont = int(input())
