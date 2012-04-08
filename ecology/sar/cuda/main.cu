// nvcc main.cu -o sar -lcuda -lm

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <ctime>

#define NICHE_F(h,u,E,mysigma)  (h*exp(-(u-E)*(u-E)/(2*mysigma*mysigma)))
#define EMPTY -1

int main(int argc, char *argv[])
{
    // Seed:
    unsigned int seed = time(NULL);

    // List of variables for run conditions:
    const int num_species = 50;
    const int num_patches = 300;
    const int num_step = 5000;
    const int search_radius = 5;

    // Species characteristics:
    const float d = 0.5; // Species mean dispersal distance (shapes dispersal kernel): 1/d
    const float m = 0.1; // Disturbance probability (disturbance kills individual)
    const float niche_min = 0.1; // Competitive strength of empty cells
    const float emi_from_out = 0.001; // Seed bank

    float *h_matrix_e = (float*)malloc(num_patches * num_patches * sizeof(float));
    int *occupied = (int*)malloc(num_patches * num_patches * sizeof(int));
    int *disturbance_counter = (int*)malloc(num_patches * num_patches * sizeof(int));

    // List of global species characteristic parameters (read in from file or defined in main function):
    float *h = (float*)malloc(num_species * sizeof(float));
    float *u = (float*)malloc(num_species * sizeof(float));
    float *c = (float*)malloc(num_species * sizeof(float));
    float *sigma = (float*)malloc(num_species * sizeof(float));

    // read in species characteristics from file
    char buffer[100];
    sprintf(buffer, "%s_st.txt", argv[1]);
    std::ifstream input_traits(buffer);
    for (int i = 0; i < num_species; ++i)
    {
        input_traits >> sigma[i]; // Niche width
        input_traits >> h[i]; // Performance at niche optimum (height, y at maximum)
        input_traits >> c[i]; // Seed production
        input_traits >> u[i]; // Resource at niche optimum (location, x at maximum)
    }
    input_traits.close();
  
    sprintf(buffer, "%s_species.txt", argv[1]);
    std::ifstream input_species(buffer);
    {
        int i = 0, j = 0;
        while (!input_species.eof())
        {
            input_species >> i;
            input_species >> j;
            input_species >> E[i][j];
            input_species >> occupied[i][j];
            input_species >> disturbance_counter[i][j];
        }
    }
    input_species.close();

    for (int block = 0; block < 1; ++block)
    {
        sprintf(buffer, "%s_dest%d.txt", argv[1], block);
        std::ifstream input_dest(buffer);
        
        for (int i = 0; i < num_patches; ++i)
        {
            for (int j = 0; j < num_patches; ++j)
            {
                disturbance_counter[i][j] = 0;
                //                occupied[i][j] = 0;
            }
        }
        
        // Removed the input_dest as a matrix and replaced by a list (see next loop).
        
        //		for (int i = 0; i < num_patches; ++i)
        //        {
        //            for (int j = 0; j < num_patches; ++j)
        //            {
        //                input_dest >> E[i][j];
        //            }
        //        }
        
        // The input file is now x,y,E with 90000 lines
        
        {
            int i = 0, j = 0;
            while (!input_dest.eof())
            {
                input_dest >> i;
                input_dest >> j;
                input_dest >> E[i][j];
            }
        }
        
        input_dest.close();
        float all_possible_seeds = 0;

        for (int dx = -search_radius; dx <= search_radius; dx++)
        {
            for (int dy = -search_radius; dy <= search_radius; dy++)
            {
                if ((dx != 0) || (dy != 0))
                {
                    all_possible_seeds += exp(-d * sqrt(dx * dx + dy * dy));
                }
            }
        }
        all_possible_seeds += emi_from_out * num_species;

        for (int t = 0; t < num_step; ++t)
        {
            for (int cell = 0; cell < num_patches * num_patches; cell++)
            {
                const int x = (int)(rng.Fixed() * num_patches);
                const int y = (int)(rng.Fixed() * num_patches);

                if(rng.Fixed() < m)
                {
                    occupied[x][y] = EMPTY;
                    ++disturbance_counter[x][y]; // BR
                }

                float Niche_res = niche_min;
                if (occupied[x][y] != EMPTY)
                {
                    int i = occupied[x][y];
                    Niche_res = NICHE_F(h[i], u[i], E[x][y], sigma[i]);
                }

                float *Seed = (float*)calloc(num_species, sizeof(float));

                const int dx_min = (x - search_radius < 0) ? -x : -search_radius; 
                const int dx_max = (x + search_radius >= num_patches) ? num_patches - 1 - x : search_radius;
                const int dy_min = (y - search_radius < 0) ? -y : -search_radius;
                const int dy_max = (y + search_radius >= num_patches) ? num_patches - 1 - y : search_radius; 

                assert(x + dx_min >= 0 && x + dx_max < num_patches);
                assert(y + dy_min >= 0 && y + dy_max < num_patches);

                for (int dx = dx_min; dx <= dx_max; ++dx)
                {
                    for (int dy = dy_min; dy <= dy_max; ++dy)
                    {
                        if (((dx != 0) || (dy != 0)) && (occupied[x + dx][y + dy] != EMPTY))
                        {
                            int i = occupied[x + dx][y + dy];
                            if (NICHE_F(h[i], u[i], E[x][y], sigma[i]) > Niche_res)
                            {
                                Seed[i] += c[i] * (exp(-d * sqrt(dx * dx + dy * dy)));
                            }
                        }
                    }
                }
                float all_seeds = 0.0;
                for (int i = 0; i < num_species; ++i)
                {
                    if (NICHE_F(h[i], u[i], E[x][y], sigma[i]) > Niche_res)
                    {
                        Seed[i] += emi_from_out;
                    }
                    all_seeds += Seed[i];
                }

                bool total_colon = false;
                if (all_seeds / all_possible_seeds > rng.Fixed())
                {
                    total_colon = true;
                }

                float *Prob_recruit = (float*)calloc(num_species, sizeof(float));
                if (total_colon == true)
                {
                    //Prob_recruit[0] = Colon[0]/double(total_colon);
                    Prob_recruit[0] = Seed[0] / all_seeds;
                    for (int i = 1; i < num_species; ++i)
                    {
                        Prob_recruit[i] = Seed[i] / all_seeds + Prob_recruit[i-1];
                    }

                    occupied[x][y] = EMPTY;

                    float randnumb = rng.Fixed();
                    for (int i = 0; i < num_species; i++)
                    {
                        if (randnumb < Prob_recruit[i])
                        {
                            assert(NICHE_F(h[i], u[i], E[x][y], sigma[i]) > niche_min);
                            occupied[x][y] = i;
                            break;
                        }
                    }
                }
                free(Seed);
                free(Prob_recruit);
            } // ends loop over cells
        } // ends loop t

        sprintf(buffer, "%s_out200_%d.txt", argv[1], block);
        std::ofstream out(buffer);
        for (int x = 0; x < num_patches; ++x)
        {
            for (int y = 0; y < num_patches; ++y)
            {
                out << x << " " << y << " " << E[x][y] << " " << occupied[x][y] << " " << disturbance_counter[x][y] << "\n";
            }
        }
        out.close();
    }
    return EXIT_SUCCESS;
}

