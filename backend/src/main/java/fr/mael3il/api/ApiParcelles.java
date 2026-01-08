package fr.mael3il.api;

import fr.mael3il.objets.Parcelle;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.Response.ResponseBuilder;
import org.eclipse.microprofile.context.ThreadContext;
import org.neo4j.driver.Driver;
import org.neo4j.driver.Values;
import org.neo4j.driver.async.AsyncSession;

import java.util.List;
import java.util.concurrent.CompletionStage;

@Path("api/v1/parcelles")
@Consumes(MediaType.APPLICATION_JSON)
@Produces(MediaType.APPLICATION_JSON)
public class ApiParcelles {

    @Inject
    Driver driver;

    @Inject
    ThreadContext threadContext;


    @GET
    public CompletionStage<Response> getAllParcelles() {
        AsyncSession session = driver.session(AsyncSession.class);
        CompletionStage<List<Parcelle>> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync("MATCH (p:Parcelle) RETURN p;")
                        .thenCompose(cursor -> cursor
                                .listAsync(record -> Parcelle.from(record.get("p").asNode()))));

        return threadContext.withContextCapture(cs)
                .thenCompose(parcelle ->
                        session.closeAsync().thenApply(signal -> parcelle))
                .thenApply(Response::ok)
                .thenApply(ResponseBuilder::build);
    }

    @GET
    @Path("/{id}")
    public CompletionStage<Response> getParcelleByID(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);
        CompletionStage<Parcelle> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync("MATCH (p:Parcelle {id: $id}) RETURN p", Values.parameters("id", id))
                        .thenCompose(cursor -> cursor
                                .singleAsync()
                                .thenApply(record -> Parcelle.from(record.get("p").asNode()))
                        ));

        return threadContext.withContextCapture(cs)
                .thenCompose(parcelle ->
                        session.closeAsync().thenApply(signal -> parcelle))
                .thenApply(parcelle -> {
                    if (parcelle == null) return Response.status(Response.Status.NOT_FOUND).build();
                    return Response.ok(parcelle).build();
                })
                .exceptionally(ex -> Response.status(Response.Status.NOT_FOUND).build());
    }

    @POST
    public CompletionStage<Response> createParcelle(Parcelle parcelle) {
        System.out.println("maison");
        AsyncSession session = driver.session(AsyncSession.class);
        CompletionStage<Parcelle> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync("CREATE (p:Parcelle {id: randomUUID(), nom: $nom, superficie: $superficie, culture: $culture, type_sol: $type_sol, latitude: $latitude, longitude: $longitude}) RETURN p",
                                Values.parameters(
                                        "nom", parcelle.getNom(),
                                        "superficie", parcelle.getSuperficie(),
                                        "culture", parcelle.getCulture(),
                                        "type_sol", parcelle.getTypeSol(),
                                        "latitude", parcelle.getLatitude(),
                                        "longitude", parcelle.getLongitude()))
                        .thenCompose(cursor -> cursor
                                .singleAsync()
                                .thenApply(record -> Parcelle.from(record.get("p").asNode()))));
        return threadContext.withContextCapture(cs)
                .thenCompose(createdParcelle ->
                        session.closeAsync().thenApply(signal -> createdParcelle))
                .thenApply(createdParcelle -> Response.status(Response.Status.CREATED).entity(createdParcelle).build());
    }

    @PUT
    @Path("/{id}")
    public CompletionStage<Response> updateParcelle(@PathParam("id") String id, Parcelle parcelle) {
        AsyncSession session = driver.session(AsyncSession.class);
        CompletionStage<Parcelle> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync("MATCH (p:Parcelle {id: $id}) SET p.nom = $nom, p.superficie = $superficie, p.culture = $culture, p.type_sol = $type_sol, p.latitude = $latitude, p.longitude = $longitude RETURN p",
                                Values.parameters(
                                        "id", id,
                                        "nom", parcelle.getNom(),
                                        "superficie", parcelle.getSuperficie(),
                                        "culture", parcelle.getCulture(),
                                        "type_sol", parcelle.getTypeSol(),
                                        "latitude", parcelle.getLatitude(),
                                        "longitude", parcelle.getLongitude()))
                        .thenCompose(cursor -> cursor
                                .singleAsync()
                                .thenApply(record -> Parcelle.from(record.get("p").asNode()))));
        return threadContext.withContextCapture(cs)
                .thenCompose(updatedParcelle ->
                        session.closeAsync().thenApply(signal -> updatedParcelle))
                .thenApply(updatedParcelle -> Response.ok(updatedParcelle).build());
    }

    @DELETE
    @Path("/{id}")
    public CompletionStage<Response> deleteParcelle(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);
        CompletionStage<Void> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync("MATCH (p:Parcelle {id: $id}) DELETE p",
                                Values.parameters("id", id))
                        .thenApply(result -> null));
        return threadContext.withContextCapture(cs)
                .thenCompose(signal ->
                        session.closeAsync().thenApply(s -> signal))
                .thenApply(signal -> Response.ok().build());
    }
}
