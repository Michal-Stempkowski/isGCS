

#if !defined(PREFERENCES_H)
#define PREFERENCES_H

template <int sentence_length, int max_symbol_length, int num_of_blocks, int num_of_threads>
class preferences
{
public:
	preferences()
	{
	}

	~preferences()
	{
	}

	int get_sentence_length() 
	{
		return sentence_length;
	}

	int get_max_symbol_length() 
	{
		return max_symbol_length;
	}

	int get_num_of_blocks() 
	{
		return num_of_blocks;
	}

	int get_num_of_threads() 
	{
		return num_of_threads;
	}

private:

};


#endif