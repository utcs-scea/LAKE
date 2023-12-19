#include <linux/list.h>

/**
 * list_search_by_id_and_add - Search a list by comparing the 'id' field of each element.
 * If a matching element is found, add a new element to the list.
 * @type: Type of the structure containing the list_head and id fields.
 * @head: Pointer to the head of the list.
 * @id_field: Name of the 'id' field in the structure.
 * @target_id: The value to search for in the 'id' field.
 * @new_element: Pointer to the new element to add if a match is found.
 * @result: Pointer to a variable that will store the result.
 *
 * Usage example:
 * struct my_data {
 *     int id;
 *     struct list_head list;
 * };
 * struct my_data *result;
 * struct my_data *new_element;
 * list_search_by_id_and_add(struct my_data, &my_list, id, 42, new_entry, result);
 */
#define list_search_by_id_and_add(type, head, id_field, target_id, new_element, result) \
    do {                                                                               \
        struct list_head *pos;                                                        \
        list_for_each(pos, head) {                                                    \
            result = list_entry(pos, type, id_field);                                 \
            if (result->id == target_id) {                                            \
                list_add(&new_element->list, &result->list.next);                    \
                break;                                                                 \
            }                                                                          \
        }                                                                              \
        if (&pos == (head)) {                                                          \
            result = NULL;                                                             \
        } else {                                                                       \
            result = new_element;                                                      \
        }                                                                              \
    } while (0)
