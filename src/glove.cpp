#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <sstream>
#include <limits>
#include <memory>
#include <cstdio>
#include <functional>

#include "args.hpp"

typedef float real;
typedef struct cooccur_rec
{
    int word1;
    int word2;
    double val;
} CREC;

class TrainingCheckPoint
{
public:
    TrainingCheckPoint(
        const std::vector<real> &weights,
        const std::vector<real> &gradsq,
        real cost) : weights(weights),
                     gradsq(gradsq),
                     cost(cost)
    {
    }

    const std::vector<real> &get_weights() { return weights; }
    const std::vector<real> &get_gradsq() { return gradsq; }
    const real get_cost() { return cost; }

private:
    std::vector<real> weights, gradsq;
    real cost;
};

class GloVe
{
public:
    GloVe(
        int vector_size,
        int num_threads,
        int load_init_param,
        int load_init_gradsq,
        const std::string &init_param_file,
        const std::string &init_gradsq_file,
        const std::string &input_file,
        const std::string &vocab_file,
        const std::string &save_W_file,
        const std::string &save_gradsq_file,
        real alpha,
        real x_max,
        real grad_clip_value,
        real eta,
        real eta_decay,
        int num_iter,
        int use_binary,
        int save_gradsq,
        int write_header,
        int model) :

                     vector_size(vector_size),
                     num_threads(num_threads),
                     load_init_param(load_init_param),
                     load_init_gradsq(load_init_gradsq),
                     init_param_file(init_param_file),
                     init_gradsq_file(init_gradsq_file),
                     input_file(input_file),
                     vocab_file(vocab_file),
                     save_W_file(save_W_file),
                     save_gradsq_file(save_gradsq_file),
                     cost(num_threads),
                     lines_per_thread(num_threads),
                     alpha(alpha),
                     x_max(x_max),
                     grad_clip_value(grad_clip_value),
                     eta(eta),
                     eta_decay(eta_decay),
                     num_iter(num_iter),
                     use_binary(use_binary),
                     save_gradsq(save_gradsq),
                     write_header(write_header),
                     model(model)
    {
        std::fstream fin(input_file, std::ios::binary | std::ios::in | std::ios::ate);
        if (!fin.is_open())
        {
            log_file_loading_error("cooccurrence file", input_file);
            std::terminate();
        }
        auto file_size = fin.tellg();
        num_lines = file_size / (sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's

        std::fstream fid(vocab_file, std::ios::in);
        if (!fid.is_open())
        {
            log_file_loading_error("vocab file", vocab_file);
            std::terminate();
        }

        std::string word;
        std::getline(fid, word);
        while (fid)
        {
            vocab_size++; // Count number of entries in vocab_file
            std::getline(fid, word);
        }
        if (vocab_size == 0)
        {
            std::cerr << "Unable to find any vocab entries in vocab file " << vocab_file << std::endl;
            std::terminate();
        }

        weights.resize(2 * vocab_size * (vector_size + 1));
        gradsq.resize(2 * vocab_size * (vector_size + 1));

        for (int a = 0; a < num_threads - 1; a++)
        {
            lines_per_thread[a] = num_lines / num_threads;
        }
        lines_per_thread[lines_per_thread.size() - 1] = num_lines / num_threads + num_lines % num_threads;
    };

private:
    int seed = 0;
    long long vocab_size = 0;
    int vector_size;
    int num_threads;
    int load_init_param, load_init_gradsq;
    std::string init_param_file, init_gradsq_file, input_file, vocab_file, save_W_file, save_gradsq_file;
    std::vector<real> weights, gradsq, cost;
    std::vector<long long> lines_per_thread;
    real alpha, x_max;
    real grad_clip_value;
    real eta;
    real eta_decay = 0.5;
    int num_iter;
    long long num_lines;
    int use_binary;
    int save_gradsq;
    int write_header;
    int model;
    std::unique_ptr<TrainingCheckPoint> last_checkpoint;

    const int log_file_loading_error(const std::string &file_description, const std::string &file_name)
    {
        std::cerr << "Unable to open " << file_description << " " << file_name << std::endl;
        std::cerr << "Errno: " << errno << std::endl;
        return errno;
    }

    int load_init_file(const std::string &file_name, std::vector<real> &array)
    {
        std::fstream fin(file_name, std::ios::binary | std::ios::in);

        if (!fin.is_open())
        {
            log_file_loading_error("init file", file_name);
            return -1;
        }

        for (size_t a = 0; a < array.size(); a++)
        {
            if (fin.eof())
            {
                std::cerr << "EOF reached before data fully loaded in "
                          << "file_name" << std::endl;
                return -1;
            }
            fin.read((char *)&array[a], sizeof(real));
        }
        return 0;
    }

    void initialize_parameters()
    {
        // TODO: return an error code when an error occurs, clean up in the calling routine
        if (seed == 0)
        {
            seed = time(0);
        }
        std::cerr << "Using random seed " << seed << std::endl;
        srand(seed);

        if (load_init_param)
        {
            // Load existing parameters
            std::cerr << std::endl
                      << "Loading initial parameters from " << init_param_file << std::endl;
            if (load_init_file(init_param_file, weights))
            {
                std::terminate();
            }
        }
        else
        {
            // Initialize new parameters
            std::for_each(weights.begin(), weights.end(), [this](auto &w)
                          { w = (rand() / (real)RAND_MAX - 0.5) / (real)vector_size; });
        }

        if (load_init_gradsq)
        {
            // Load existing squared gradients
            std::cerr << std::endl
                      << "Loading initial squared gradients from " << init_gradsq_file << std::endl;
            if (load_init_file(init_gradsq_file, gradsq))
            {
                std::terminate();
            }
        }
        else
        {
            // Initialize new squared gradients
            std::for_each(gradsq.begin(), gradsq.end(), [this](auto &g)
                          { g = 1.0; });
        }
    }

    std::string build_filename(const std::string &prefix, const std::string &ext, int nb_iter)
    {
        if (nb_iter < 0)
            return prefix + "." + ext;
        else
            return prefix + "." + std::to_string(nb_iter) + "." + ext;
    }

    void open_and_check(std::fstream &fs, const std::string &filename, std::ios_base::openmode mode)
    {
        fs.open(filename, mode);
        if (!fs.is_open())
        {
            log_file_loading_error("file", save_W_file);
            std::terminate();
        }
    }

    void save_params(int nb_iter)
    {
        /*
        * nb_iter is the number of iteration (= a full pass through the cooccurrence matrix).
        *   nb_iter  > 0 => checkpointing the intermediate parameters, so nb_iter is in the filename of output file.
        *   nb_iter == 0 => checkpointing the initial parameters
        *   else         => saving the final paramters, so nb_iter is ignored.
        */

        std::string output_file, output_file_gsq;
        std::string word;
        std::fstream fid, fout;
        std::fstream fgs;

        if (use_binary > 0 || nb_iter == 0)
        {
            // Save parameters in binary file
            // note: always save initial parameters in binary, as the reading code expects binary
            output_file = build_filename(save_W_file, "bin", nb_iter);
            open_and_check(fout, output_file, std::ios::binary | std::ios::out);
            for (auto &w : weights)
            {
                fout.write((char *)&w, sizeof(real));
            }
            fout.close();

            if (save_gradsq > 0)
            {
                output_file_gsq = build_filename(save_gradsq_file, "bin", nb_iter);
                open_and_check(fgs, output_file_gsq, std::ios::binary | std::ios::out);
                for (auto &g : gradsq)
                {
                    fgs.write((char *)&g, sizeof(real));
                }
                fgs.close();
            }
        }
        if (use_binary != 1)
        { // Save parameters in text file
            output_file = build_filename(save_W_file, "txt", nb_iter);
            open_and_check(fout, output_file, std::ios::binary | std::ios::out);

            if (save_gradsq > 0)
            {
                output_file_gsq = build_filename(save_gradsq_file, "txt", nb_iter);
                open_and_check(fgs, output_file_gsq, std::ios::binary | std::ios::out);
            }

            open_and_check(fid, vocab_file, std::ios::in);

            if (write_header)
            {
                fout << vocab_size << " " << vector_size << std::endl;
            }

            for (long long a = 0; a < vocab_size; a++)
            {
                std::string fid_line;
                std::getline(fid, fid_line);
                std::stringstream fidss(fid_line);

                fidss >> word;
                if (word == "<unk>")
                {
                    std::terminate();
                }

                fout << word;
                if (model == 0)
                { // Save all parameters (including bias)
                    for (int b = 0; b < (vector_size + 1); b++)
                        fout << " " << weights[a * (vector_size + 1) + b];

                    for (int b = 0; b < (vector_size + 1); b++)
                        fout << " " << (weights[(vocab_size + a) * (vector_size + 1) + b]);
                }
                if (model == 1) // Save only "word" vectors (without bias)
                    for (int b = 0; b < vector_size; b++)
                        fout << " " << weights[a * (vector_size + 1) + b];
                if (model == 2) // Save "word + context word" vectors (without bias)
                    for (int b = 0; b < vector_size; b++)
                        fout << " " << (weights[a * (vector_size + 1) + b] + weights[(vocab_size + a) * (vector_size + 1) + b]);
                fout << std::endl;
                if (save_gradsq > 0)
                { // Save gradsq
                    fgs << word;

                    for (int b = 0; b < (vector_size + 1); b++)
                        fgs << " " << gradsq[a * (vector_size + 1) + b];
                    for (int b = 0; b < (vector_size + 1); b++)
                        fgs << " " << (gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                    fgs << std::endl;
                }
            }

            std::vector<real> unk_vec(vector_size + 1);
            std::vector<real> unk_context(vector_size + 1);
            word = "<unk>";

            long long num_rare_words = vocab_size < 100 ? vocab_size : 100;

            for (long long a = vocab_size - num_rare_words; a < vocab_size; a++)
            {
                for (int b = 0; b < (vector_size + 1); b++)
                {
                    unk_vec[b] += weights[a * (vector_size + 1) + b] / (real)num_rare_words;
                    unk_context[b] += weights[(vocab_size + a) * (vector_size + 1) + b] / (real)num_rare_words;
                }
            }

            fout << word;
            if (model == 0)
            { // Save all parameters (including bias)
                for (int b = 0; b < (vector_size + 1); b++)
                    fout << " " << unk_vec[b];
                for (int b = 0; b < (vector_size + 1); b++)
                    fout << " " << unk_context[b];
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (int b = 0; b < vector_size; b++)
                    fout << " " << unk_vec[b];
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (int b = 0; b < vector_size; b++)
                    fout << " " << (unk_vec[b] + unk_context[b]);
            fout << std::endl;
        }

        if (nb_iter > 0)
        { // remove previous param files
            auto prev_iter = nb_iter - 1;
            std::remove(build_filename(save_W_file, "bin", prev_iter).c_str());
            std::remove(build_filename(save_W_file, "txt", prev_iter).c_str());
            std::remove(build_filename(save_gradsq_file, "bin", prev_iter).c_str());
            std::remove(build_filename(save_gradsq_file, "txt", prev_iter).c_str());
        }
    }

    const real check_nan(real update)
    {
        if (std::isnan(update) || std::isinf(update))
        {
            std::cerr << std::endl
                      << "caught NaN in update";
            return 0.;
        }
        else
        {
            return update;
        }
    }

    void glove_thread(int id)
    {
        CREC cr;
        real diff, fdiff, temp1, temp2;
        std::fstream fin(input_file, std::ios::binary | std::ios::in);

        if (!fin.is_open())
        {
            // TODO: exit all the threads or somehow mark that glove failed
            log_file_loading_error("input file", input_file);
            return;
        }
        fin.seekg((num_lines / num_threads * id) * (sizeof(CREC)), std::ios::beg);
        cost[id] = 0;

        std::vector<real> W_updates1(vector_size);
        std::vector<real> W_updates2(vector_size);

        for (auto a = 0; a < lines_per_thread[id]; a++)
        {
            std::size_t l1, l2;

            fin.read((char *)&cr, sizeof(CREC));

            if (fin.eof())
            {
                break;
            }
            if (cr.word1 < 1 || cr.word2 < 1)
            {
                continue;
            }

            /* Get location of words in W & gradsq */
            l1 = (cr.word1 - 1LL) * (vector_size + 1);                // cr word indices start at 1
            l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words

            /* Calculate cost, save diff for gradients */
            diff = 0;
            for (int b = 0; b < vector_size; b++)
            {
                diff += weights[b + l1] * weights[b + l2]; // dot product of word and context word vector
            }
            diff += weights[vector_size + l1] + weights[vector_size + l2] - std::log(cr.val); // add separate bias for each word
            fdiff = (cr.val > x_max) ? diff : std::pow(cr.val / x_max, alpha) * diff;         // multiply weighting function (f) with diff

            // Check for NaN and inf() in the diffs.
            if (std::isnan(diff) || std::isnan(fdiff) || std::isinf(diff) || std::isinf(fdiff))
            {
                std::cerr << "Caught NaN in diff for kdiff for thread. Skipping update";
                continue;
            }

            cost[id] += 0.5 * fdiff * diff; // weighted squared error

            /* Adaptive gradient updates */
            real W_updates1_sum = 0;
            real W_updates2_sum = 0;
            for (int b = 0; b < vector_size; b++)
            {
                // learning rate times gradient for word vectors
                temp1 = std::fmin(std::fmax(fdiff * weights[b + l2], -grad_clip_value), grad_clip_value) * eta;
                temp2 = std::fmin(std::fmax(fdiff * weights[b + l1], -grad_clip_value), grad_clip_value) * eta;
                // adaptive updates
                W_updates1[b] = temp1 / std::sqrt(gradsq[b + l1]);
                W_updates2[b] = temp2 / std::sqrt(gradsq[b + l2]);
                W_updates1_sum += W_updates1[b];
                W_updates2_sum += W_updates2[b];
                gradsq[b + l1] += temp1 * temp1;
                gradsq[b + l2] += temp2 * temp2;
            }
            if (!std::isnan(W_updates1_sum) && !std::isinf(W_updates1_sum) && !std::isnan(W_updates2_sum) && !std::isinf(W_updates2_sum))
            {
                for (int b = 0; b < vector_size; b++)
                {
                    weights[b + l1] -= W_updates1[b];
                    weights[b + l2] -= W_updates2[b];
                }
            }

            // updates for bias terms
            weights[vector_size + l1] -= check_nan(fdiff / std::sqrt(gradsq[vector_size + l1]));
            weights[vector_size + l2] -= check_nan(fdiff / std::sqrt(gradsq[vector_size + l2]));
            fdiff *= fdiff;
            gradsq[vector_size + l1] += fdiff;
            gradsq[vector_size + l2] += fdiff;
        }
    }

    std::string current_time()
    {
        time_t rawtime;
        struct tm *info;
        char time_buffer[80];

        time(&rawtime);
        info = localtime(&rawtime);
        strftime(time_buffer, 80, "%x - %I:%M.%S%p", info);

        return std::string(time_buffer);
    }

public:
    /* Train model */
    void train()
    {
        last_checkpoint.reset(
            new TrainingCheckPoint(
                weights,
                gradsq,
                std::numeric_limits<real>::infinity()));

        std::cerr << "TRAINING MODEL" << std::endl;

        std::cerr << "Read " << num_lines << " lines." << std::endl;
        std::cerr << "Initializing parameters...";
        initialize_parameters();
        std::cerr << "done." << std::endl;
        std::cerr << "vector size: " << vector_size << std::endl;
        std::cerr << "vocab size: " << vocab_size << std::endl;
        std::cerr << "x_max: " << x_max << std::endl;
        std::cerr << "alpha: " << alpha << std::endl;
        std::cerr << "epochs: " << num_iter << std::endl;
        std::cerr << "eta: " << eta << std::endl;

        // Lock-free asynchronous SGD
        for (int b = 0; b < num_iter; b++)
        {
            std::vector<std::thread> pt;
            real total_cost = 0;

            for (int a = 0; a < num_threads; a++)
            {
                pt.emplace_back(std::thread([this](int id)
                                            { glove_thread(id); },
                                            a));
            }

            for (int a = 0; a < num_threads; a++)
            {
                pt[a].join();
            }

            for (int a = 0; a < num_threads; a++)
            {
                total_cost += cost[a];
            }

            std::cerr << current_time() << ", iter: " << (b + 1) << ", cost: " << (total_cost / (real)num_lines) << std::endl;
            if (checkpoint(total_cost))
            {
                --b;
            }

            save_params(b + 1);
        }
        save_params(-1);
    }

    bool checkpoint(real cost)
    {
        if (cost > (*last_checkpoint).get_cost())
        {
            return restore_last_checkpoint();
        }
        else
        {
            return save_checkpoint(cost);
        }
    }

    bool save_checkpoint(real cost)
    {
        last_checkpoint.reset(new TrainingCheckPoint(weights, gradsq, cost));
        return false;
    }

    bool restore_last_checkpoint()
    {
        weights = (*last_checkpoint).get_weights();
        gradsq = (*last_checkpoint).get_gradsq();
        std::cerr << current_time() << " cost increased, restoring last training checkpoint and lowering eta from " << eta;
        eta *= eta_decay;
        std::cerr << " to " << eta << std::endl;
        return true;
    }
};

int main(int argc, char **argv)
{
    int vector_size = 600;
    int num_threads = 8;
    int load_init_param = 0;
    int load_init_gradsq = 0;
    std::string init_param_file = "vectors.000.bin";
    std::string init_gradsq_file = "gradsq.000.bin";
    std::string input_file = "cooccurrence.shuf.bin";
    std::string vocab_file = "vocab.txt";
    std::string save_W_file = "vectors";
    std::string save_gradsq_file = "gradsq";
    real alpha = 0.75;
    real x_max = 100.0;
    real grad_clip_value = 100.0;
    real eta = 0.05;
    real eta_decay = 0.5;
    int num_iter = 100;
    int use_binary = 2;
    int save_gradsq = 0;
    int write_header = 0;
    int model = 2;

    std::vector<std::shared_ptr<Parameter> > parameters = {
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-write-header", "int", "0", "If 1, write vocab_size/vector_size as first line. Do nothing if 0 (default).", write_header)),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-vector-size", "int", "600", "Dimension of word vector representations (excluding bias term)", vector_size)),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-threads", "int", "8", "Number of threads", num_threads)),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-iter", "int", "100", "Number of training iterations", num_iter)),
        std::shared_ptr<Parameter>(new TemplateParameter<float>("-eta", "float", "0.05", "Initial learning rate", eta)),
        std::shared_ptr<Parameter>(new TemplateParameter<float>("-eta-decay", "float", "0.5", "Learning rate decay when epoch loss increases", eta_decay)),
        std::shared_ptr<Parameter>(new TemplateParameter<float>("-alpha", "float", "0.75", "Parameter in exponent of weighting function", alpha)),
        std::shared_ptr<Parameter>(new TemplateParameter<float>("-x-max", "float", "100.0", "Parameter specifying cutoff in weighting function", x_max)),
        std::shared_ptr<Parameter>(new TemplateParameter<float>("-grad-clip", "float", "100.0", "Gradient components clipping parameter. Values will be clipped to [-grad-clip, grad-clip] interval", grad_clip_value)),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-binary", "int", "2", "Save output in binary format (0: text, 1: binary, 2: both)", use_binary)),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-model", "int", "2", "Model for word vector output (for text output only)\n\t\t   0: output all data, for both word and context word vectors, including bias terms\n\t\t   1: output word vectors, excluding bias terms\n\t\t   2: output word vectors + context word vectors, excluding bias terms", model)),
        std::shared_ptr<Parameter>(new TemplateParameter<std::string>("-input-file", "file", "cooccurrence.shuf.bin", "Binary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle')", input_file)),
        std::shared_ptr<Parameter>(new TemplateParameter<std::string>("-vocab-file", "file", "vocab.txt", "File containing vocabulary (truncated unigram counts, produced by 'vocab_count')", vocab_file)),
        std::shared_ptr<Parameter>(new TemplateParameter<std::string>("-save-file", "file", "vectors", "Filename, excluding extension, for word vector output", save_W_file)),
        std::shared_ptr<Parameter>(new TemplateParameter<std::string>("-gradsq-file", "file", "gradsq", "Filename, excluding extension, for squared gradient output", save_gradsq_file, [&save_gradsq](){save_gradsq=1;})),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-save-gradsq", "int", "0", "Save accumulated squared gradients; ignored if gradsq-file is specified", save_gradsq)),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-load-init-param", "int", "0", "Load initial parameters from -init-param-file", load_init_param)),
        std::shared_ptr<Parameter>(new TemplateParameter<std::string>("-init-param-file", "file", "vectors.000.bin", "Binary initial parameters file to be loaded if -load-init-params is 1", init_param_file)),
        std::shared_ptr<Parameter>(new TemplateParameter<int>("-load-init-gradsq", "int", "0", "Load initial squared gradients from -init-gradsq-file", load_init_gradsq)),
        std::shared_ptr<Parameter>(new TemplateParameter<std::string>("-init-gradsq-file", "file", "gradsq.000.bin", "Binary initial squared gradients file to be loaded if -load-init-gradsq is 1", init_gradsq_file))
    };

    if (argc == 1)
    {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        print_parameters(parameters);
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
    }
    else
    {

        load_parameters(parameters, argc, argv);

        GloVe glove(
            vector_size,
            num_threads,
            load_init_param,
            load_init_gradsq,
            init_param_file,
            init_gradsq_file,
            input_file,
            vocab_file,
            save_W_file,
            save_gradsq_file,
            alpha,
            x_max,
            grad_clip_value,
            eta,
            eta_decay,
            num_iter,
            use_binary,
            save_gradsq,
            write_header,
            model);

        glove.train();
    }
}
