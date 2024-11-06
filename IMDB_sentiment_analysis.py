#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>

struct Node {
    int value;
    struct Node *next;
    struct Node *prev;
};

// List struct to maintain head and tail for the linked list
struct List {
    struct Node *head;
    struct Node *tail;
    int size;
    int max_size; // The maximum number of nodes allowed (K)
};

// Function to create a new node
struct Node *create_node(int value) {
    struct Node *node = malloc(sizeof(struct Node));
    if (node == NULL) {
        fprintf (stderr, "%s: Couldn't create memory for the node; %s\n", "linkedlist", strerror(errno));
    exit(-1);
  }
    node->value = value;
    node->next = NULL;
    node->prev = NULL;
    return node;
}

// Function to create an empty list with a max size
struct List *create_list() {
  struct List *list = malloc(sizeof(struct List));
  if (list == NULL) {
    fprintf (stderr, "%s: Couldn't create memory for the list; %s\n", "linkedlist", strerror (errno));
    exit(-1);
  }
  list->head = NULL;
  list->tail = NULL;
  return list;
}

// Function to check if a value already exists in the list
int find_by_value(struct List *list, int value) {
    struct Node *current = list->head;
    while (current != NULL) {
        if (current->value == value) {
            return 1; // Value found
        }
        current = current->next;
    }
    return 0; // Value not found
}

// Insert a value in sorted order (largest to smallest) in the list
void insert_sorted(struct List *list, int value) {
    if (find_by_value(list, value)) {
        return; // Skip if value already exists
    }

    struct Node *newNode = create_node(value);

    // Insert at the beginning if empty or if new value is largest
    if (list->head == NULL || value > list->head->value) {
        newNode->next = list->head;
        if (list->head != NULL) {
            list->head->prev = newNode;
        }
        list->head = newNode;
        if (list->tail == NULL) {
            list->tail = newNode;
        }
    } else {
        // Insert in sorted position
        struct Node *current = list->head;
        while (current->next != NULL && current->next->value > value) {
            current = current->next;
        }
        newNode->next = current->next;
        if (current->next != NULL) {
            current->next->prev = newNode;
        }
        current->next = newNode;
        newNode->prev = current;
        if (newNode->next == NULL) {
            list->tail = newNode;
        }
    }

    list->size++;

    // Remove the last element if size exceeds max_size
    if (list->size > list->max_size) {
        struct Node *toDelete = list->tail;
        list->tail = list->tail->prev;
        if (list->tail != NULL) {
            list->tail->next = NULL;
        }
        free(toDelete);
        list->size--;
    }
}

// Function to process each integer in a file and add to the list
void process_file(const char *filepath, struct List *list) {
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open file %s\n", filepath);
        return;
    }

    int value;
    while (fscanf(file, "%d", &value) != EOF) {
        insert_sorted(list, value);
    }

    fclose(file);
}

// Process all files in the given directory
void process_directory(const char *directoryPath, struct List *list) {
    DIR *dir = opendir(directoryPath);
    if (dir == NULL) {
        fprintf(stderr, "Could not open directory %s\n", directoryPath);
        exit(1);
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.' || entry->d_name[strlen(entry->d_name) - 1] == '~') {
            continue; // Skip hidden and temporary files
        }
        
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", directoryPath, entry->d_name);
        process_file(filepath, list);
    }

    closedir(dir);
}

// Print the list (for debugging)
void print_list(struct List *list) {
    struct Node *ptr = list->head;
    while (ptr != NULL) {
        printf("%d ", ptr->value);
        ptr = ptr->next;
    }
    printf("\n");
}

// Write the list contents to an output file
void write_output(const char *outputFile, struct List *list) {
    FILE *file = fopen(outputFile, "w");
    if (file == NULL) {
        fprintf(stderr, "Could not open output file %s\n", outputFile);
        exit(1);
    }

    struct Node *current = list->head;
    while (current != NULL) {
        fprintf(file, "%d\n", current->value);
        current = current->next;
    }

    fclose(file);
}

// Function to destroy the list and free memory
void destroy_list(struct List *list) {
    struct Node *current = list->head;
    while (current != NULL) {
        struct Node *toDelete = current;
        current = current->next;
        free(toDelete);
    }
    free(list);
}

// Main function to demonstrate the code
int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <K> <directoryPath> <outputFile>\n", argv[0]);
        return 1;
    }

    int K = atoi(argv[1]);
    const char *directoryPath = argv[2];
    const char *outputFile = argv[3];

    // Create the list with max size K
    struct List *list = create_list(K);

    // Process directory and add top K integers
    process_directory(directoryPath, list);

    // Write the top K integers to output file
    write_output(outputFile, list);

    // Clean up memory
    destroy_list(list);

    return 0;
}
