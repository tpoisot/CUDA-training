// g++ -Irng -O3 main.cpp rng/Random.cpp -o sar

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <ctime>
#include <RandomLib/Random.hpp>

#define NICHE_F(h,u,E,mysigma)  (h*exp(-(u-E)*(u-E)/(2*mysigma*mysigma)))
#define EMPTY -1

typedef std::vector<std::vector<double> > array2Ddouble;
typedef std::vector<std::vector<int> > array2Dint;

void sar(char *argv[], unsigned int seed);
template<typename Data> void SetArraySize2d(std::vector<std::vector<Data > >& , int, int);

int main(int argc, char *argv[])
{
    const time_t t0 = time(NULL);

    // Random number generator
    unsigned int seed = t0;

    // Run the model
    sar(argv, seed);

    std::cout << "Time elasped: " << (time(NULL) - t0) << " seconds." << std::endl;
    return 0;
}

void **mat_alloc(int nrows, int ncols, size_t size)
{
    void **matrix = (void**)malloc(nrows * sizeof(void*));

    int i = 0;
    for (; i < nrows; ++i)
    {
        matrix[i] = malloc(ncols * size);
    }
    return matrix;
}

void mat_free(void **mat, int nrows)
{
    int i = 0;
    for (; i < nrows; ++i)
    {
        free(mat[i]);
    }
    free(mat);
} 

template<typename Data> void SetArraySize2d(std::vector<std::vector<Data > > &matrix, int xsize, int ysize)
{
    matrix.resize(xsize);
    for (int i = 0; i < xsize; ++i)
    {
        matrix[i].resize(ysize);
    }
}

void sar(char *argv[], unsigned int seed)
{
    RandomLib::Random rng;
    rng.Reseed();

    // List of variables for run conditions:
    const int num_species = 50;
    const int num_patches = 300;
    const int num_step = 5000;
    const int search_radius = 5;

    // Species characteristics:
    const double d = 0.5; // Species mean dispersal distance (shapes dispersal kernel): 1/d
    const double m = 0.1; // Disturbance probability (disturbance kills individual)
    const double niche_min = 0.1; // Competitive strength of empty cells
    const double emi_from_out = 0.001; // Seed bank

    array2Ddouble E;
    array2Dint occupied;
    array2Dint disturbance_counter;

    SetArraySize2d(E, num_patches, num_patches);
    SetArraySize2d(occupied, num_patches, num_patches);
    SetArraySize2d(disturbance_counter, num_patches, num_patches);

    // List of global species characteristic parameters (read in from file or defined in main function):
    double *h = new double[num_species];
    double *u = new double[num_species];
    double *c = new double[num_species];
    double *sigma = new double[num_species];

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
		
		//        for (int x = 0; x < num_patches; x++)
		//        {
		//            for (int y = 0; y < num_patches; y++)
		//            {
		//                const int s = (int)(rng.Fixed() * num_species); // Select a species at random
		//                occupied[x][y] = NICHE_F(h[s], u[s], E[x][y], sigma[s]) < niche_min ? EMPTY : s;
		//            }
		//       }
        double all_possible_seeds = 0;

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

                double Niche_res = niche_min;
                if (occupied[x][y] != EMPTY)
                {
                    int i = occupied[x][y];
                    Niche_res = NICHE_F(h[i], u[i], E[x][y], sigma[i]);
                }

                std::vector<double> Seed(num_species, 0.0);

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
                double all_seeds = 0.0;
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

                std::vector<double> Prob_recruit(num_species, 0.0);
                if (total_colon == true)
                {
                    //Prob_recruit[0] = Colon[0]/double(total_colon);
                    Prob_recruit[0] = Seed[0] / all_seeds;
                    for (int i = 1; i < num_species; ++i)
                    {
                        Prob_recruit[i] = Seed[i] / all_seeds + Prob_recruit[i-1];
                    }

                    occupied[x][y] = EMPTY;

                    double randnumb = rng.Fixed();
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
}
