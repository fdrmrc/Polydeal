#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <algorithm>

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>()
  {}

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double> &          values,
             const unsigned int /*component*/) const override;
};

template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p,
           const unsigned int component = 0) const override;
};

template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double> &          values,
                               const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = 8. * numbers::PI * numbers::PI *
                std::sin(2. * numbers::PI * points[i][0]) *
                std::sin(2. * numbers::PI * points[i][1]);
}


template <int dim>
double
Solution<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);
}

template <int dim>
Tensor<1, dim>
Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  Assert(false, ExcNotImplemented());
  return return_value;
}


template <int dim>
class TestIterator
{
private:
  void
  make_grid();
  void
  setup_agglomeration();
  void
  test1();
  void
  test2();
  void
  test3();



  Triangulation<dim>                         tria;
  MappingQ<dim>                              mapping;
  FE_DGQ<dim>                                dg_fe;
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;

public:
  TestIterator();
  void
  run();
};



template <int dim>
TestIterator<dim>::TestIterator()
  : mapping(1)
  , dg_fe(1)
{
  std::cout << "dim = " << dim << std::endl;
}

template <int dim>
void
TestIterator<dim>::make_grid()
{
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(6);
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
}



template <int dim>
void
TestIterator<dim>::setup_agglomeration()

{
  // std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
  //   3, 9}; //{8, 9, 10, 11};
  std::vector<types::global_cell_index> idxs_to_be_agglomerated = {
    3235, 3238}; //{3,9};

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated,
                                             cells_to_be_agglomerated);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated2 = {
    831, 874}; //{25,19}

  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated2;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated2,
                                             cells_to_be_agglomerated2);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated3 = {1226, 1227};
  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated3;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated3,
                                             cells_to_be_agglomerated3);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated4 = {
    2279, 2278}; //{36,37}
  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated4;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated4,
                                             cells_to_be_agglomerated4);


  std::vector<types::global_cell_index> idxs_to_be_agglomerated5 = {
    3760, 3761}; //{3772,3773}
  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated5;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated5,
                                             cells_to_be_agglomerated5);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated6 = {3648, 3306};
  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated6;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated6,
                                             cells_to_be_agglomerated6);

  std::vector<types::global_cell_index> idxs_to_be_agglomerated7 = {3765, 3764};
  std::vector<typename Triangulation<dim>::active_cell_iterator>
    cells_to_be_agglomerated7;
  PolyUtils::collect_cells_for_agglomeration(tria,
                                             idxs_to_be_agglomerated7,
                                             cells_to_be_agglomerated7);


  // Agglomerate the cells just stored
  ah = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
  ah->insert_agglomerate(cells_to_be_agglomerated);
  ah->insert_agglomerate(cells_to_be_agglomerated2);
  ah->insert_agglomerate(cells_to_be_agglomerated3);
  ah->insert_agglomerate(cells_to_be_agglomerated4);
  ah->insert_agglomerate(cells_to_be_agglomerated5);
  ah->insert_agglomerate(cells_to_be_agglomerated6);
  ah->insert_agglomerate(cells_to_be_agglomerated7);
  ah->distribute_agglomerated_dofs(dg_fe);
}



template <int dim>
void
TestIterator<dim>::test1()
{
  const unsigned int                   dofs_per_cell = ah->n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  auto                                 polytope = ah->begin();
  {
    for (; polytope != ah->end(); ++polytope)
      {
        const unsigned int n_faces = polytope->n_faces();
        std::cout << "n_faces =" << n_faces << std::endl;
        polytope->get_dof_indices(local_dof_indices);
        std::cout << "Global DoF indices for polytope " << polytope->index()
                  << std::endl;
        for (const types::global_dof_index idx : local_dof_indices)
          std::cout << idx << std::endl;
      } // Loop over polytopes
  }
  std::cout << std::endl;
  std::cout << "Looping backwards: " << std::endl;
  {
    // Looping from the end
    auto polytope = ah->last();
    for (; polytope != ah->begin(); --polytope)
      {
        const unsigned int n_faces = polytope->n_faces();
        std::cout << "n_faces =" << n_faces << std::endl;
        polytope->get_dof_indices(local_dof_indices);
        std::cout << "Global DoF indices for polytope " << polytope->index()
                  << std::endl;
        for (const types::global_dof_index idx : local_dof_indices)
          std::cout << idx << std::endl;
      } // Loop over polytopes
  }
}



template <int dim>
void
TestIterator<dim>::test2()
{
  std::cout << "Test with IteratorRange" << std::endl;

  const unsigned int                   dofs_per_cell = ah->n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto &polytope : ah->polytope_iterators())
    {
      const unsigned int n_faces = polytope->n_faces();
      std::cout << "n_faces =" << n_faces << std::endl;
      polytope->get_dof_indices(local_dof_indices);
      std::cout << "Global DoF indices for polytope " << polytope->index()
                << std::endl;
      for (const types::global_dof_index idx : local_dof_indices)
        std::cout << idx << std::endl;
    }
}



template <int dim>
void
TestIterator<dim>::test3()
{
  std::cout << "Test polytope_boundary()" << std::endl;

  const unsigned int                   dofs_per_cell = ah->n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto &polytope : ah->polytope_iterators())
    {
      std::cout << "Polytope number: " << polytope->index()
                << "\t volume: " << polytope->volume() << std::endl;
      const auto & boundary   = polytope->polytope_boundary();
      unsigned int face_index = 0;
      for (const auto &face : boundary)
        {
          std::cout << "Face index: " << ++face_index << std::endl;
          for (unsigned int vertex_index :
               GeometryInfo<dim - 1>::vertex_indices())
            std::cout << face->vertex(vertex_index) << std::endl;
        }
      std::cout << std::endl;
    }
}



template <int dim>
void
TestIterator<dim>::run()
{
  make_grid();
  setup_agglomeration();
  test1();
  test2();
  test3();
}

int
main()
{
  TestIterator<2> test;
  test.run();

  return 0;
}