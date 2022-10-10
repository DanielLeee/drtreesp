/* 
 * Minimum directed spanning trees with worst-case complexity O(n^2)
 * Rewritten based on Miloš Stanojević's code: https://github.com/stanojevic/Fast-MST-Algorithm/blob/main/mst.py
 * By Yeshu Li
 */
#include "stdio.h"
#include "float.h"
#include "math.h"
#include "stdbool.h"
#include "stdlib.h"
#include "string.h"


typedef struct edge_priority_queue
{
    int *target;
    double *weight;
    bool *mask;
} epq;


void epq_construct(epq *q, double **weight, int n, int node_id)
{
    q->target = malloc(sizeof(*q->target) * n);
    q->weight = malloc(sizeof(*q->weight) * n);
    q->mask = malloc(sizeof(*q->mask) * n);

    memset(q->mask, 1, sizeof(*q->mask) * n);
    q->mask[node_id] = false;

    for(int i = 0; i < n; ++i)
    {
        q->target[i] = node_id;
        q->weight[i] = weight[i][node_id];
    }

    return;
}


void epq_destruct(epq *q)
{
    free(q->target);
    free(q->weight);
    free(q->mask);

    return;
}


void epq_extract_min(epq *q, int n, int *dst_u, int *dst_v, double *dst_min_val)
{
    int min_idx = -1;

    for(int i = 0; i < n; ++i)
        if(q->mask[i] && (min_idx == -1 || q->weight[i] < q->weight[min_idx]))
            min_idx = i;

    if(min_idx == -1)
    {
        *dst_u = -1;
    }
    else
    {
        q->mask[min_idx] = false;
        *dst_u = min_idx;
        *dst_v = q->target[min_idx];
        *dst_min_val = q->weight[min_idx];
    }

    return;
}


void epq_meld(epq *q_dst, epq *q_src, int n)
{
    for(int i = 0; i < n; ++i)
        if(!q_src->mask[i])
            q_dst->mask[i] = false;
        else if(q_src->weight[i] < q_dst->weight[i])
        {
            q_dst->target[i] = q_src->target[i];
            q_dst->weight[i] = q_src->weight[i];
        }
        

    return;
}


void epq_add_const(epq *q, int n, double c)
{
    for(int i = 0; i < n; ++i)
        if(q->mask[i])
            q->weight[i] += c;
    
    return;
}


int dsu_find(int *dsu, int x)
{
    return dsu[x] == x ? x : (dsu[x] = dsu_find(dsu, dsu[x]));
}


void dismantle(int *parent, int **children, int *cq, int u)
{
    while(parent[u] != u)
    {
        for(int i = 1, v; i <= children[parent[u]][0]; ++i)
        {
            v = children[parent[u]][i];
            if(v == u)
                continue;
            parent[v] = v;
            if(children[v][0] > 0)
                cq[++cq[0]] = v;
        }
        u = parent[u];
    }

    return;
}


void tarjan(double **weight, int *result, int n)
{
    int nn = 2 * n - 1;
    double *in_w = malloc(sizeof(*in_w) * nn);
    int *in_u = malloc(sizeof(*in_u) * nn);
    int *in_v = malloc(sizeof(*in_v) * nn);
    int *prev = malloc(sizeof(*prev) * nn);
    int **children = malloc(sizeof(*children) * (nn));
    int *parent = malloc(sizeof(*parent) * nn);
    int *dsu = malloc(sizeof(*dsu) * nn);
    int *cq = malloc(sizeof(*cq) * n);
    epq *q = malloc(sizeof(*q) * nn);
    int new_node, cur_node, next_node, u, v, x, next_free = n;
    double w;

    cq[0] = 0;
    memset(in_u, -1, sizeof(*in_u) * nn);
    for(int i = 0; i < n; ++i)
        epq_construct(&q[i], weight, n, i);
    for(int i = 0; i < nn; ++i)
    {
        children[i] = malloc(sizeof(*children[i]) * (n + 1));
        children[i][0] = 0;
        parent[i] = dsu[i] = i;
    }


    cur_node = n - 1;
    while(true)
    {
        epq_extract_min(&q[cur_node], n, &u, &v, &w);
        if(u == -1)
            break;
        
        in_u[cur_node] = u;
        in_v[cur_node] = v;
        in_w[cur_node] = w;
        next_node = dsu_find(dsu, u);
        prev[cur_node] = next_node;
        
        if(in_u[u] == -1)
            cur_node = next_node;
        else
        {
            new_node = next_free++;
            
            x = cur_node;
            do
            {
                children[new_node][++children[new_node][0]] = x;
                x = dsu_find(dsu, prev[x]);
            }
            while(x != cur_node);
            
            for(int i = 1, ci; i <= children[new_node][0]; ++i)
            {
                ci = children[new_node][i];
                parent[ci] = new_node;
                dsu[ci] = new_node;
                epq_add_const(&q[ci], n, -in_w[ci]);
                if(i == 1)
                    q[new_node] = q[ci];
                else
                    epq_meld(&q[new_node], &q[ci], n);
            }

            cur_node = new_node;
        }
    }

    dismantle(parent, children, cq, 0);
    for(int i = 1; i <= cq[0]; ++i)
    {
        cur_node = cq[i];
        in_u[in_v[cur_node]] = in_u[cur_node];
        dismantle(parent, children, cq, in_v[cur_node]);
    }
    for(int i = 1; i < n; ++i)
        result[i] = in_u[i];

    free(in_u);
    free(in_v);
    free(in_w);
    free(prev);
    for(int i = 0; i < nn; ++i)
        free(children[i]);
    for(int i = 0; i < n; ++i)
        epq_destruct(&q[i]);
    free(children);
    free(q);
    free(parent);
    free(dsu);
    free(cq);

    return;
}


bool is_tree(int *proposal, int n)
{
    bool *visited = malloc(sizeof(*visited) * n), okay = true;
    int *q = malloc(sizeof(*q) * n), qn = 1;

    memset(visited, 0, sizeof(*visited) * n);
    q[0] = 0;

    for(int i = 0; i < qn; ++i)
    {
        if(visited[q[i]])
        {
            okay = false;
            break;
        }
        visited[q[i]] = true;
        for(int j = 0; j < n; ++j)
            if(proposal[j] == q[i])
                q[qn++] = j;
    }

    okay &= (qn == n);
    
    free(visited);
    free(q);
    
    return okay;
}


void fast_parse(double *para_weight, bool one_root, int32_t n, int32_t num_rels, int32_t *result_3d)
{
    int tot_size = n * n * num_rels;
    int root_count = 0;
    double w_min = DBL_MAX, w_max = DBL_MIN, correction;
    double **weight = malloc(sizeof(*weight) * n);
    int **rels = malloc(sizeof(*rels) * n);
    int *result = malloc(sizeof(*result) * n);

    for(int i = 0, offset = 0; i < n; ++i)
    {
        weight[i] = malloc(sizeof(*weight[i]) * n);
        rels[i] = malloc(sizeof(*rels[i]) * n);
        memset(rels[i], 0, sizeof(*rels[i]) * n);
        for(int j = 0, r; j < n; ++j, offset += num_rels)
        {
            r = 0;
            for(int k = 1; k < num_rels; ++k)
                if(para_weight[offset + k] < para_weight[offset + r])
                    r = k;
            weight[i][j] = para_weight[offset + r];
            rels[i][j] = r;
        }
    }

    memset(result, 0, sizeof(*result) * n);

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
        {
            if(weight[i][j] < w_min)
                w_min = weight[i][j];
            if(weight[i][j] > w_max)
                w_max = weight[i][j];
            if(weight[i][j] < weight[result[j]][j])
                result[j] = i;
        }
    
    result[0] = -1;
    for(int i = 1; i < n; ++i)
        root_count += (result[i] == 0);

    if(is_tree(result, n) && (!one_root || root_count == 1))
        ;
    else
    {
        if(one_root)
        {
            correction = (n - 1) * (w_max - w_min) + 1;
            for(int i = 1; i < n; ++i)
            {
                weight[0][i] += correction;
                if(weight[0][i] > w_max)
                    w_max = weight[0][i];
            }
        }
        correction = w_max + fabs(w_max) + 1;
        for(int i = 1; i < n; ++i)
            weight[i][0] = correction;
        tarjan(weight, result, n);
    }

    memset(result_3d, 0, sizeof(*result_3d) * tot_size);
    for(int i = 1; i < n; ++i)
        result_3d[result[i] * n * num_rels + i * num_rels + rels[result[i]][i]] = 1;

    for(int i = 0; i < n; ++i)
        free(weight[i]), free(rels[i]);
    free(weight);
    free(rels);
    free(result);
    
    return;
}

